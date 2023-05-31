import hydra.utils
import ipdb
import tqdm
import wandb
import contextlib
import pathlib
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import einops

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import disentangle


@eqx.filter_jit
def train_step(model, optimizer, optimizer_state, data, step, key):
    (loss, aux), grads = eqx.filter_value_and_grad(model.batched_loss, has_aux=True)(model, data, step, key=key)
    grads = disentangle.utils.optax(grads)
    model = disentangle.utils.optax(model)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
    model = eqx.apply_updates(model, updates)
    model = disentangle.utils.unoptax(model)

    for x, name in zip([disentangle.utils.unoptax(grads), disentangle.utils.unoptax(updates)], ['grads', 'updates']):
        aux['metrics'].update({
            f'{name}_{k_partition}/{k_norm}': v
            for k_norm, v_norm in x.partition_norms().items() for k_partition, v in v_norm.items()
        })
    try:
        aux['metrics']['weight_decay'] = model.weight_decay(optimizer_state)
    except:
        pass

    return model, optimizer_state, aux


def log_disentanglement_metrics(model, aux, step):
    pairwise_mutual_informations = {}
    mask_hats = {}
    normalizations = ['sources']
    if model.latent.is_continuous:
        for normalization in normalizations:
            estimator = 'discrete-discrete'
            for bins in [10, 20, 50, 'auto']:
                name = f'normalization={normalization}/{estimator}/bins={bins}'
                pairwise_mutual_informations[name], mask_hats[name] = disentangle.metrics.compute_pairwise_mutual_information(
                    latents=aux['z_hat'],
                    sources=aux['z_true'],
                    continuous_latents=model.latent.is_continuous,
                    estimator=estimator,
                    bins=bins,
                    normalization=normalization
                )
            estimator = 'continuous-discrete'
            name = f'normalization={normalization}/{estimator}'
            pairwise_mutual_informations[name], mask_hats[name] = disentangle.metrics.compute_pairwise_mutual_information(
                latents=aux['z_hat'],
                sources=aux['z_true'],
                continuous_latents=model.latent.is_continuous,
                estimator=estimator,
                normalization=normalization
            )
    else:
        for normalization in normalizations:
            estimator = 'discrete-discrete'
            name = f'normalization={normalization}/{estimator}'
            pairwise_mutual_informations[name], mask_hats[name] = disentangle.metrics.compute_pairwise_mutual_information(
                latents=aux['z_hat'],
                sources=aux['z_true'],
                continuous_latents=model.latent.is_continuous,
                estimator=estimator,
                normalization=normalization
            )

    for k, v in pairwise_mutual_informations.items():
        fig, ax = plt.subplots(figsize=(6, aux['z_hat'].shape[1] ** 0.8))
        sns.heatmap(v, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1, yticklabels=[f'z{i}' for i in range(v.shape[0])], xticklabels=[f's{i}' for i in range(v.shape[1])], annot_kws={'fontsize': 8})
        for i, label in enumerate(ax.get_yticklabels()):
            if mask_hats[k][i] == 0:
                label.set_color('red')
        fig.tight_layout()
        wandb.log({f'pairwise_mutual_information/{k}': wandb.Image(fig)}, step=step)
        plt.close()

        for info_metric in ['modularity', 'compactness']:
            wandb.log({
                f'{info_metric}/{k}/ratio': disentangle.metrics.compute_mutual_information_ratio(v, mask_hats[k], info_metric),
                f'{info_metric}/{k}/gap': disentangle.metrics.compute_mutual_information_gap(v, mask_hats[k], info_metric)

            }, step=step)

    wandb.log({
        f'explicitness_classification': disentangle.metrics.explicitness_classification(aux['z_hat'], aux['z_true']),
        f'explicitness_regression': disentangle.metrics.explicitness_regression(aux['z_hat'], aux['z_true'])
    }, step=step)


def log_latent_densities(aux, step):
    data = pd.DataFrame(aux['z_hat'], columns=[f'z{i}' for i in range(aux['z_hat'].shape[1])])
    data['id'] = data.index
    data = data.melt(id_vars='id', var_name='z', value_name='value')
    fig, ax = plt.subplots(figsize=(aux['z_hat'].shape[1] ** 0.8, 3))
    sns.violinplot(data=data, ax=ax, kind='violin', x='z', y='value', scale='width', cut=0)
    fig.tight_layout()
    wandb.log({f'latents/density': wandb.Image(fig)}, step=step)
    plt.close()


def log_reconstruction_metrics(aux, step):
    num_samples = 16
    true = einops.rearrange(aux['x_true'][:num_samples], 'b c h w -> h (b w) c')
    predicted = einops.rearrange(jax.nn.sigmoid(aux['x_hat_logits'][:num_samples]), 'b c h w -> h (b w) c')
    absolute_diff = jnp.abs(true - predicted)
    image = jnp.concatenate([true, predicted, absolute_diff], axis=0)
    psnr = jax.vmap(disentangle.metrics.peak_signal_to_noise_ratio)(aux['x_hat_logits'], aux['x_true'])
    wandb.log({
        f'reconstructions': wandb.Image(np.asarray(image)),
        f'peak_signal_to_noise_ratio': psnr.mean().item()
    }, step=step)


def evaluate(model, val_set, config, step, key):
    iterate_key, sample_key = jax.random.split(key)

    auxs = []
    for batch in iter(val_set):
        iterate_key, sub_key = jax.random.split(iterate_key)
        _, aux = model.batched_loss(model, batch, step, key=sub_key)
        aux['x_true'] = batch['x']
        aux['z_true'] = batch['z']
        if isinstance(model, disentangle.models.VQVAE):
            aux['z_hat'] = aux['outs']['z_indices']
        else:
            aux['z_hat'] = aux['outs']['z_hat']
        aux['x_hat_logits'] = aux['outs']['x_hat_logits']
        auxs.append(aux)

    aux = disentangle.utils.transpose_and_gather(auxs)

    wandb.log({f'{k}/val': v.mean().item() for k, v in aux['metrics'].items()}, step=step)

    log_reconstruction_metrics(aux, step)
    log_disentanglement_metrics(model, aux, step)
    log_latent_densities(aux, step)

    if not isinstance(model, disentangle.models.VQVAE):
        latent_mins = aux['z_hat'].min(axis=0)
        latent_maxs = aux['z_hat'].max(axis=0)

        # generations and perturbations
        num_samples = 16
        num_perturbations = 16
        for i_latent in range(model.latent.num_latents):
            latent_perturbed = jnp.tile(aux['z_hat'][:num_samples], (num_perturbations, 1, 1))    # (num_perturbations, num_samples, num_latents)
            latent_perturbed = latent_perturbed.at[:, :, i_latent].set(jnp.linspace(latent_mins[i_latent], latent_maxs[i_latent], num_perturbations)[:, None])
            x = []
            for i_perturbation in range(num_perturbations):
                x.append(
                    jax.nn.sigmoid(
                        jax.vmap(model.decoder)(latent_perturbed[i_perturbation])['x_hat_logits']
                    )
                )
            x = jnp.stack(x)
            image = einops.rearrange(x, 'v n c h w -> (n h) (v w) c')
            wandb.log({f'latent {i_latent} perturbations': wandb.Image(np.asarray(image))}, step=step)


def save(path, model, optimizer_state):
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / 'model.eqx', model)
    eqx.tree_serialise_leaves(path / 'optimizer_state.eqx', optimizer_state)
    print(f'saved model and optimizer state to {path}')


def load(path, model, optimizer_state):
    model = eqx.tree_deserialise_leaves(path / 'model.eqx', model)
    optimizer_state = eqx.tree_deserialise_leaves(path / 'optimizer_state.eqx', optimizer_state)
    return model, optimizer_state


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
    model = hydra.utils.instantiate(config.model_partial)(dataset_size=dataset_metadata['num_train'], x=batch['x'][0], key=model_key)

    optimizer, optimizer_state = model.construct_optimizer(config)

    pbar = tqdm.tqdm(train_set, total=int(config.optim.num_steps))

    context = jax.disable_jit if config.debug else contextlib.nullcontext
    with context():
        for step, batch in enumerate(pbar):
            if step >= config.optim.num_steps:
                break

            if (step + 1) % config.checkpoint.period == 0:
                path = checkpoint_base / f'step={step}'
                save(path, model, optimizer_state)
                wandb.save(str(path / '*'), base_path=run.dir)

            if step == 0 or (step + 1) % config.eval.period == 0 or ((step + 1 < config.eval.period) and (step + 1) % (config.eval.period // 10) == 0):
                eval_key, key = jax.random.split(eval_key)
                model = model.eval()
                evaluate(model, val_set, config, jnp.array(step), key)

            train_key, step_key = jax.random.split(train_key)
            model = model.train()
            model, optimizer_state, aux = train_step(model, optimizer, optimizer_state, batch, jnp.array(step), step_key)
            wandb.log({f'{k}/train': v.mean().item() for k, v in aux['metrics'].items()}, step=step)
