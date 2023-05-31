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

# Set the global font to be Times New Roman, size 10 (or any other size you want)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# Use LaTeX to handle the mathematics in your figure
plt.rcParams['mathtext.fontset'] = 'stix'  # 'stix' font includes math symbols
plt.rcParams['font.serif'] = 'Times New Roman'

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

    normalization = 'sources'
    if model.latent.is_continuous:
        mi, mask_hats = disentangle.metrics.compute_pairwise_mutual_information(
            latents=aux['z_hat'],
            sources=aux['z_true'],
            continuous_latents=model.latent.is_continuous,
            estimator='continuous-discrete',
            normalization=normalization
        )
    else:
        mi, mask_hats = disentangle.metrics.compute_pairwise_mutual_information(
            latents=aux['z_hat'],
            sources=aux['z_true'],
            continuous_latents=model.latent.is_continuous,
            estimator='discrete-discrete',
            normalization=normalization
        )

    fig, ax = plt.subplots(figsize=(3, 6))
    # sns.heatmap(mi, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1,
    #             yticklabels=[rf'$\mathbf{{z}}_{i}$' for i in range(mi.shape[0])], xticklabels=[f'\mathbf{{s}}_{i}' for i in range(mi.shape[1])],
    #             annot_kws={'fontsize': 8})
    sns.heatmap(mi, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1, cbar=True, annot_kws={'fontsize': 8},
                yticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(mi.shape[0])], xticklabels=[rf'$\mathbf{{s}}_{{{i}}}$' for i in range(mi.shape[1])],
                rasterized=True)
    for i, label in enumerate(ax.get_yticklabels()):
        if mask_hats[i] == 0:
            label.set_color('red')
    fig.tight_layout()
    # plt.savefig('/iris/u/kylehsu/code/disentangle/vis/paper/quantized_ae_shapes3d.svg')
    # plt.savefig('/iris/u/kylehsu/code/disentangle/vis/paper/tcvae_shapes3d.svg')
    # plt.savefig('/iris/u/kylehsu/code/disentangle/vis/paper/ae_shapes3d.svg')
    plt.savefig('/iris/u/kylehsu/code/disentangle/vis/paper/ae_shapes3d_cbar.svg')

    ipdb.set_trace()
    # plt.savefig()
    plt.close()


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
    run_name = f'plot_mi__{model_name}__{data_name}'
    run = disentangle.utils.initialize_wandb(config, name=run_name)
    checkpoint_base = pathlib.Path(run.dir) / 'checkpoint'

    data_key = jax.random.PRNGKey(config.data.seed)
    model_key, loader_key, train_key, val_key = jax.random.split(jax.random.PRNGKey(config.experiment.seed), 4)

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

    model_file = wandb.restore(f'checkpoint/step={config.load.step}/model.eqx', run_path=config.load.run_path)
    model = eqx.tree_deserialise_leaves(model_file.name, model)

    model = model.eval()
    evaluate(model, val_set, config, 0, jax.random.PRNGKey(0))