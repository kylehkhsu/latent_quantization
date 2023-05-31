import ipdb
import jax
import jax.numpy as jnp
import scipy
import sklearn.metrics
import sklearn.feature_selection
from sklearn.metrics import mutual_info_score
import numpy as np


def discretize_unique(z):
    """

    :param z: (num_samples, num_dims)
    :return:
    """
    ret = np.zeros_like(z, dtype=np.int32)
    for i in range(z.shape[1]):
        unique, labels = np.unique(z[:, i], return_inverse=True)
        ret[:, i] = labels
    return ret


def discretize_binning(z, bins):
    ret = np.zeros_like(z, dtype=np.int32)
    for i in range(z.shape[1]):
        ret[:, i] = np.digitize(z[:, i], np.histogram(z[:, i], bins=bins)[1][:-1])
    return ret


def compute_entropy(z):
    """

    :param z: (num_samples, num_dims)
    :return:
    """
    ret = np.zeros(z.shape[1])
    if z.dtype in [np.int32, np.int64, jnp.int32, jnp.int64]:
        for i in range(z.shape[1]):
            ret[i] = sklearn.metrics.mutual_info_score(z[:, i], z[:, i])
    else:
        for i in range(z.shape[1]):
            ret[i] = sklearn.feature_selection.mutual_info_regression(z[:, i][:, None], z[:, i])
    return ret


def compute_mig(latents, sources, continuous_latents):
    mi, latent_mask = compute_pairwise_mutual_information(
        latents, sources, continuous_latents=continuous_latents, estimator='discrete-discrete', bins=20, normalization='sources'
    )
    return compute_mutual_information_gap(mi, latent_mask, per='source')


def compute_pairwise_mutual_information(latents, sources, continuous_latents, estimator='continuous-discrete', bins=20, normalization='sources'):
    ret = np.zeros((latents.shape[1], sources.shape[1]))
    sources = discretize_unique(sources)
    source_entropies = compute_entropy(sources)

    latent_ranges = jnp.max(latents, axis=0) - jnp.min(latents, axis=0)
    if continuous_latents:
        latent_mask = latent_ranges > jnp.max(latent_ranges) / 8
        match estimator:
            case 'continuous-discrete':
                pass
            case 'discrete-discrete':
                latents = discretize_binning(latents, bins)
            case _:
                raise ValueError
    else:
        # latent_mask = latent_ranges > 0
        latent_mask = latent_ranges > jnp.max(latent_ranges) / 8
        latents = discretize_unique(latents)
    latent_entropies = compute_entropy(latents)

    for i in range(latents.shape[1]):
        for j in range(sources.shape[1]):
            match estimator:
                case 'continuous-discrete':
                    ret[i, j] = sklearn.feature_selection.mutual_info_classif(latents[:, i][:, None], sources[:, j])
                case 'discrete-discrete':
                    ret[i, j] = sklearn.metrics.mutual_info_score(latents[:, i], sources[:, j])
                case _:
                    raise ValueError

    match normalization:
        case 'geometric_mean':
            ret /= np.sqrt(latent_entropies[:, None] * source_entropies[None, :])
        case 'arithmetic_mean':
            ret /= ((latent_entropies[:, None] + source_entropies[None, :]) / 2)
        case 'sources':
            ret /= source_entropies[None, :]
        case 'latents':
            ret /= latent_entropies[:, None]
        case 'none':
            pass
        case _:
            raise ValueError

    ret = jnp.nan_to_num(ret, nan=0.)
    return ret, latent_mask


def compute_mutual_information_gap(pairwise_mutual_information, latent_mask, info_metric):
    """

    :param pairwise_mutual_information: (num_latents, num_sources)
    :param latent_mask: (num_latents,)
    :param per: 'latent' -> modularity, 'source' -> compactness
    :return:
    """

    match info_metric:
        case 'modularity':
            mig = np.zeros(pairwise_mutual_information.shape[0])
            sorted_pairwise_mutual_information = np.sort(pairwise_mutual_information, axis=1)
            for i in range(pairwise_mutual_information.shape[0]):
                mig[i] = (sorted_pairwise_mutual_information[i, -1] - sorted_pairwise_mutual_information[i, -2])
            mig = mig[latent_mask]
        case 'compactness':
            mig = np.zeros(pairwise_mutual_information.shape[1])
            sorted_pairwise_mutual_information = np.sort(pairwise_mutual_information, axis=0)
            for i in range(pairwise_mutual_information.shape[1]):
                mig[i] = (sorted_pairwise_mutual_information[-1, i] - sorted_pairwise_mutual_information[-2, i])
        case _:
            raise ValueError
    return jnp.mean(mig)


def compute_mutual_information_ratio(pairwise_mutual_information, latent_mask, info_metric):
    pairwise_mutual_information = pairwise_mutual_information * latent_mask[:, None]
    num_sources = pairwise_mutual_information.shape[1]
    num_active_latents = jnp.sum(latent_mask)

    match info_metric:
        case 'modularity':
            preferences = jnp.max(pairwise_mutual_information, axis=1) / jnp.sum(pairwise_mutual_information, axis=1)
            mask = jnp.isfinite(preferences)
            mir = (jnp.sum(preferences[mask]) / jnp.sum(mask) - 1 / num_sources) / (1 - 1 / num_sources)
        case 'compactness':
            preferences = jnp.max(pairwise_mutual_information, axis=0) / jnp.sum(pairwise_mutual_information, axis=0)
            mir = (jnp.sum(preferences) / num_sources - 1 / num_active_latents) / (1 - 1 / num_active_latents)
        case _:
            raise ValueError
    return mir

