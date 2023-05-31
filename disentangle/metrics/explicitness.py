import ipdb
import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax

from sklearn import linear_model, preprocessing


def process_latents(latents):
    if latents.dtype in [np.int64, np.int32, jnp.int64, jnp.int32]:
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
        latents = one_hot_encoder.fit_transform(latents)
    elif latents.dtype in [np.float32, np.float64, jnp.float32, jnp.float64]:
        standardizer = preprocessing.StandardScaler()
        latents = standardizer.fit_transform(latents)
    else:
        raise ValueError(f'latents.dtype {latents.dtype} not supported')
    return latents


def linear_regression(X, y):
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.float32, np.float64]

    model = linear_model.LinearRegression(
        fit_intercept=True,
        n_jobs=-1,
        positive=False
    )

    model.fit(X, y)
    y_hat = model.predict(X)

    n = X.shape[0]
    variance = 1 / 2

    average_negative_log_likelihood = -1 / 2 * jnp.log(2 * jnp.pi * variance) - 1 / (2 * variance) * jnp.mean(
        jnp.square(y - y_hat))
    return average_negative_log_likelihood, model.score(X, y)


@functools.cache
def marginal_entropy_regression(labels: tuple, null_shape):
    labels = jnp.array(labels)
    null = jnp.zeros(null_shape)
    return linear_regression(null, labels)


def explicitness_regression(latents, sources):
    standardizer = preprocessing.StandardScaler()
    sources = standardizer.fit_transform(sources)
    latents = process_latents(latents)

    normalized_predictive_information_per_source = []

    for i_source in range(sources.shape[1]):
        source = sources[:, i_source]

        predictive_conditional_entropy, coefficient_of_determination = linear_regression(latents, source)
        # marginal_source_entropy, _ = marginal_entropy_regression(tuple(source), standardized_latents.shape)
        #
        # normalized_predictive_information_per_source.append(
        #     (marginal_source_entropy - predictive_conditional_entropy) / marginal_source_entropy
        # )

        # equivalent to result from Prop. 1.5 in https://arxiv.org/pdf/2002.10689.pdf
        # normalization via Prop. 1.3 from the same
        normalized_predictive_information_per_source.append(coefficient_of_determination)

    return jnp.mean(jnp.array(normalized_predictive_information_per_source))


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
    logits = model.predict_log_proba(X)
    average_cross_entropy = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    return average_cross_entropy


@functools.cache
def marginal_entropy_classification(labels: tuple, null_shape):
    labels = jnp.array(labels)
    null = jnp.zeros(null_shape)
    return logistic_regression(null, labels)


def explicitness_classification(latents, sources):
    label_encoder = preprocessing.LabelEncoder()
    latents = process_latents(latents)

    normalized_predictive_information_per_source = []

    for i_source in range(sources.shape[1]):
        source = sources[:, i_source]
        labels = label_encoder.fit_transform(source)

        predictive_conditional_entropy = logistic_regression(latents, labels)
        marginal_source_entropy = marginal_entropy_classification(tuple(labels), latents.shape)

        normalized_predictive_information_per_source.append(
            (marginal_source_entropy - predictive_conditional_entropy) / marginal_source_entropy
        )

        # # sanity check
        # standardized_sources = standardizer.fit_transform(sources)
        # model.fit(standardized_sources, labels)
        # probs = model.predict_proba(standardized_sources)
        # zero1 = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.log(probs), labels))
        # zero2 = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.log(one_hot_encoder.fit_transform(labels[:, None])), labels))
        # assert jnp.isclose(zero1, zero2) and jnp.isclose(zero1, 0.0)

    return jnp.mean(jnp.array(normalized_predictive_information_per_source))


def _test():
    import omegaconf
    import disentangle

    config = omegaconf.OmegaConf.create(
        {
            'data': {
                'possible_dirs': [
                    '/scr-ssd/kylehsu/data',
                    '/scr/kylehsu/data',
                    '/iris/u/kylehsu/data',
                ],
                'seed': 42,
                'num_val_data': 10000,
                'batch_size': 10000
            },
        }
    )

    _, _, val_set = disentangle.datasets.shapes3d.get_datasets(config)

    data = next(iter(val_set))
    sources = data['z']

    for noise in jnp.linspace(0.0, 1.0, 11):
        latents = sources + jax.random.normal(jax.random.PRNGKey(42), shape=sources.shape) * noise
        print(f'noise={noise}: explicitness_reg={explicitness_regression(latents, sources)}, explicitness_clf={explicitness_classification(latents, sources)}')

    print(f'noise=inf : explicitness_reg={explicitness_regression(jnp.zeros(sources.shape), sources)}, explicitness_clf={explicitness_classification(jnp.zeros(sources.shape), sources)}')


if __name__ == '__main__':
    _test()
