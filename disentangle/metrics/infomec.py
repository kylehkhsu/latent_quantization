import numpy as np
from sklearn import preprocessing, feature_selection, metrics, linear_model


def process_sources(sources):
    processed_sources = []
    for i in range(sources.shape[1]):
        processed_sources.append(preprocessing.LabelEncoder().fit_transform(sources[:, i]))
    return np.stack(processed_sources, axis=1)


def compute_nmi(sources, latents, discrete_latents):
    processed_sources = process_sources(sources)

    processed_latents = []
    if discrete_latents:
        for j in range(latents.shape[1]):
            processed_latents.append(preprocessing.LabelEncoder().fit_transform(latents[:, j]))
        processed_latents = np.stack(processed_latents, axis=1)
    else:
        for j in range(latents.shape[1]):
            processed_latents.append(preprocessing.StandardScaler().fit_transform(latents[:, j][:, None]))
        processed_latents = np.concatenate(processed_latents, axis=1)

    ret = np.empty(shape=(processed_sources.shape[1], processed_latents.shape[1]))
    for i in range(processed_sources.shape[1]):
        for j in range(processed_latents.shape[1]):
            if discrete_latents:
                ret[i, j] = metrics.mutual_info_score(processed_sources[:, i], processed_latents[:, j])
            else:
                ret[i, j] = feature_selection.mutual_info_classif(processed_latents[:, j][:, None],
                                                                  processed_sources[:, i], discrete_features=False,
                                                                  n_neighbors=10)

        entropy = metrics.mutual_info_score(processed_sources[:, i], processed_sources[:, i])
        ret[i, :] /= entropy
    return ret


def compute_infoe(sources, latents, discrete_latents):
    normalized_predictive_information_per_source = []
    processed_sources = process_sources(sources)

    if discrete_latents:
        processed_latents = preprocessing.OneHotEncoder().fit_transform(latents)
    else:
        processed_latents = preprocessing.StandardScaler().fit_transform(latents)

    for i in range(processed_sources.shape[1]):
        predictive_conditional_entropy = logistic_regression(processed_latents, processed_sources[:, i])
        null = np.zeros_like(latents)
        marginal_source_entropy = logistic_regression(null, processed_sources[:, i])

        normalized_predictive_information_per_source.append(
            (marginal_source_entropy - predictive_conditional_entropy) / marginal_source_entropy
        )

    return np.mean(normalized_predictive_information_per_source)


def logistic_regression(X, y):
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.int32, np.int64]

    model = linear_model.LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=100,
        multi_class='multinomial',
        n_jobs=-1,
    )

    model.fit(X, y)
    y_pred = model.predict_proba(X)
    return metrics.log_loss(y, y_pred)


def compute_infomec(sources, latents, discrete_latents):
    """
    :param sources: (num_samples, num_sources)
    :param latents: (num_samples, num_latents)
    :param discrete_latents: bool
    :return: dict
    """
    nmi = compute_nmi(sources, latents, discrete_latents)

    latent_ranges = np.max(latents, axis=0) - np.min(latents, axis=0)
    if discrete_latents:
        active_latents = latent_ranges > 0
    else:
        active_latents = latent_ranges > np.max(latent_ranges) / 20

    num_sources = sources.shape[1]
    num_active_latents = np.sum(active_latents)
    pruned_nmi = nmi[:, active_latents]

    infom = (np.mean(np.max(pruned_nmi, axis=0) / np.sum(pruned_nmi, axis=0)) - 1 / num_sources) / (1 - 1 / num_sources)
    infoc = (np.mean(np.max(pruned_nmi, axis=1) / np.sum(pruned_nmi, axis=1)) - 1 / num_active_latents) / (1 - 1 / num_active_latents)

    infoe = compute_infoe(sources, latents, discrete_latents)

    return {
        'infom': infom,
        'infoe': infoe,
        'infoc': infoc,
        'nmi': nmi,
        'active_latents': active_latents,
    }
