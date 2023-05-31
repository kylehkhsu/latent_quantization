import hydra.utils
import ipdb
import tqdm
import wandb
import contextlib
import itertools
import pathlib
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import collections
from typing import Tuple, Optional, Callable, Dict
import pandas as pd
import numpy as np
import einops

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import disentangle
from .train_ae import log_disentanglement_metrics, log_latent_densities, log_reconstruction_metrics


@eqx.filter_jit
def generator_step(model, optimizers, optimizer_states, data, *, key):
    (_, generator_aux), grads = eqx.filter_value_and_grad(model.batched_generator_loss, has_aux=True)(model, data, key=key)
    grads = model.optax(grads.generator, grads.latent)
    generator_and_latent = model.optax(model.generator, model.latent)
    updates, optimizer_states['generator'] = optimizers['generator'].update(
        grads, optimizer_states['generator'], generator_and_latent
    )
    generator_and_latent = eqx.apply_updates(generator_and_latent, updates)
    generator, latent = model.unoptax(generator_and_latent)
    model = eqx.tree_at(lambda x: x.generator, model, replace=generator)
    model = eqx.tree_at(lambda x: x.latent, model, replace=latent)
    for (x_generator, x_latent), tree_name in zip([model.unoptax(grads), model.unoptax(updates)], ['grads', 'updates']):
        for x, x_name in zip([x_generator, x_latent], ['generator', 'latent']):
            generator_aux['metrics'].update({
                f'{x_name}_{tree_name}_{k_partition}/{k_norm}': v
                for k_norm, v_norm in model.partition_norms_attr(x, x_name).items() for k_partition, v in v_norm.items()
            })
    return model, optimizer_states, generator_aux

@eqx.filter_jit
def encoder_step(model, optimizers, optimizer_states, data, *, key):
    (_, encoder_aux), grads = eqx.filter_value_and_grad(model.batched_encoder_loss, has_aux=True)(model, data, key=key)
    grads = disentangle.utils.optax(grads.encoder)
    encoder = disentangle.utils.optax(model.encoder)
    updates, optimizer_states['encoder'] = optimizers['encoder'].update(
        grads, optimizer_states['encoder'], encoder
    )
    encoder = eqx.apply_updates(encoder, updates)
    encoder = disentangle.utils.unoptax(encoder)
    model = eqx.tree_at(lambda x: x.encoder, model, replace=encoder)
    for x, name in zip([disentangle.utils.unoptax(grads), disentangle.utils.unoptax(updates)], ['grads', 'updates']):
        encoder_aux['metrics'].update({
            f'encoder_{name}_{k_partition}/{k_norm}': v
            for k_norm, v_norm in model.partition_norms_attr(x, 'encoder').items() for k_partition, v in v_norm.items()
        })
    return model, optimizer_states, encoder_aux


def evaluate(model, val_set, config, step, *, key):
    auxs = []
    for batch in iter(val_set):
        outs = jax.vmap(model.encode)(batch['x'])
        aux = {
            'x_true': batch['x'],
            'z_true': batch['z'],
            'z_hat': outs['z_hat'],
        }
        outs = jax.vmap(model.generator)(outs['z_hat'])
        aux['x_hat_logits'] = outs['x_hat_logits']
        auxs.append(aux)

    aux = disentangle.utils.transpose_and_gather(auxs)
    log_reconstruction_metrics(aux, step)
    log_disentanglement_metrics(model, aux, step)
    log_latent_densities(aux, step)

    latent_mins = aux['z_hat'].min(axis=0)
    latent_maxs = aux['z_hat'].max(axis=0)

    # generations and perturbations
    num_samples = 16
    fixed_generation_key = jax.random.PRNGKey(152)
    batch = next(iter(val_set))
    _, perturb_aux = model.batched_generator_loss(model, batch, key=fixed_generation_key)
    image = einops.rearrange(perturb_aux['outs']['x_fake'][:num_samples], 'b c h w -> h (b w) c')
    wandb.log({f'unconditional generations': wandb.Image(np.asarray(image))}, step=step)

    num_perturbations = 16
    for i_latent in range(model.latent.num_latents):
        latent_perturbed = jnp.tile(perturb_aux['outs']['z_fake'][:num_samples], (num_perturbations, 1, 1))    # (num_perturbations, num_samples, num_latents)
        latent_perturbed = latent_perturbed.at[:, :, i_latent].set(jnp.linspace(latent_mins[i_latent], latent_maxs[i_latent], num_perturbations)[:, None])
        x = []
        for i_perturbation in range(num_perturbations):
            x.append(
                jax.nn.sigmoid(
                    jax.vmap(model.generator)(latent_perturbed[i_perturbation])['x_hat_logits']
                )
            )
        x = jnp.stack(x)
        image = einops.rearrange(x, 'v n c h w -> (n h) (v w) c')
        wandb.log({f'latent {i_latent} perturbations': wandb.Image(np.asarray(image))}, step=step)


def save(path, model, optimizer_states):
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / 'model.eqx', model)
    eqx.tree_serialise_leaves(path / 'optimizer_states.eqx', optimizer_states)
    print(f'saved model and optimizer state to {path}')


def load(path, model, optimizer_states):
    model = eqx.tree_deserialise_leaves(path / 'model.eqx', model)
    optimizer_states = eqx.tree_deserialise_leaves(path / 'optimizer_states.eqx', optimizer_states)
    return model, optimizer_states


def main(config):
    if config.model_partial.latent_partial.num_latents == 'twice_sources':
        match config.data.name:
            case 'shapes3d':
                config.model_partial.latent_partial.num_latents = 12
            case 'isaac3d':
                config.model_partial.latent_partial.num_latents = 18
            case 'falcor3d':
                config.model_partial.latent_partial.num_latents = 14
            case 'mpi3d':
                config.model_partial.latent_partial.num_latents = 14
            case _:
                raise ValueError(f'unknown dataset {config.data.name}')

    model_name = config.model_partial._target_.split('.')[-1]
    data_name = config.data.name
    num_latents = config.model_partial.latent_partial.num_latents
    encoder_name = config.model_partial.encoder_partial._target_.split('.')[-1]
    decoder_name = config.model_partial.decoder_partial._target_.split('.')[-1]

    if config.model_partial.encoder_partial.get('conv_partial') is not None:
        match len(config.model_partial.encoder_partial.conv_partial.widths):
            case 4:
                size = 'small'
            case 12:
                size = 'large'
            case _:
                raise ValueError('unknown model size')
    else:
        widths = config.model_partial.encoder_partial.dense_partial.widths
        size = f'{widths[0]}x{len(widths)}'

    run_name = f'size={size}|data={data_name}|model={model_name}|num_latents={num_latents}|encoder={encoder_name}|decoder={decoder_name}'
    run = disentangle.utils.initialize_wandb(config, name=run_name)
    checkpoint_base = pathlib.Path(run.dir) / 'checkpoint'

    data_key = jax.random.PRNGKey(config.data.seed)
    model_key, loader_key, train_key, eval_key = jax.random.split(jax.random.PRNGKey(config.experiment.seed), 4)

    match config.data.name:
        case 'dsprites':
            dataset_metadata, train_set, val_set = disentangle.datasets.dsprites.get_datasets(config)
        case 'shapes3d':
            dataset_metadata, train_set, val_set = disentangle.datasets.shapes3d.get_datasets(config)
        case 'isaac3d':
            dataset_metadata, train_set, val_set = disentangle.datasets.isaac3d.get_datasets(config)
        case 'falcor3d':
            dataset_metadata, train_set, val_set = disentangle.datasets.falcor3d.get_datasets(config)
        case 'mpi3d':
            dataset_metadata, train_set, val_set = disentangle.datasets.mpi3d.get_datasets(config)
        case _:
            raise ValueError(f'unknown dataset {config.data.name}')
    batch = next(iter(train_set))
    model = hydra.utils.instantiate(config.model_partial)(x=batch['x'][0], key=model_key)

    optimizers, optimizer_states = model.construct_optimizer(config)

    pbar = tqdm.tqdm(train_set, total=int(config.optim.num_steps))

    context = jax.disable_jit if config.debug else contextlib.nullcontext
    with context():
        for step, batch in enumerate(pbar):
            if step >= config.optim.num_steps:
                break

            if (step + 1) % config.checkpoint.period == 0:
                path = checkpoint_base / f'step={step}'
                save(path, model, optimizer_states)
                wandb.save(str(path / '*'), base_path=run.dir)

            if step == 0 \
                    or (step + 1) % config.eval.period == 0 \
                    or (
                        (step + 1 < config.eval.period) and
                        (step + 1) % (config.eval.period // 10) == 0
            ):
                eval_key, key = jax.random.split(eval_key)
                model = model.eval()
                evaluate(model, val_set, config, step, key=key)

            train_key, encoder_key, generator_key = jax.random.split(train_key, 3)
            model = model.train()
            model, optimizer_states, encoder_aux = encoder_step(model, optimizers, optimizer_states, batch, key=encoder_key)
            wandb.log({f'{k}/train': v.mean().item() for k, v in encoder_aux['metrics'].items()}, step=step)

            if (step + 1) % config.optim.generator_step_period == 0:
                model, optimizer_states, generator_aux = generator_step(model, optimizers, optimizer_states, batch, key=generator_key)
                wandb.log({f'{k}/train': v.mean().item() for k, v in generator_aux['metrics'].items()}, step=step)
