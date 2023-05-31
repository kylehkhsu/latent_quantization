import ipdb
import pathlib
import jax.numpy as jnp
import numpy as np
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_data(data):
    ret = {}
    x = data['image']   # (64, 64, 3)
    x = tf.transpose(x, [2, 0, 1])   # (3, 64, 64)
    x = tf.cast(x, tf.float32) / 255.
    ret['x'] = x
    ret['z'] = tf.stack([
        data['value_floor_hue'],    # 10
        data['value_object_hue'],   # 10
        data['value_orientation'],  # 10
        data['value_scale'],        # 8
        data['value_shape'],        # 4
        data['value_wall_hue'],     # 15
    ])
    return ret


def get_datasets(config):
    possible_dirs = config.data.possible_dirs
    while len(possible_dirs) > 0:
        possible_dir = pathlib.Path(possible_dirs.pop(0))
        try:
            builder = tfds.builder('shapes3d', data_dir=possible_dir)
            builder.download_and_prepare()
            break
        except PermissionError as e:
            print(e)
    metadata = {
        'num_train': builder.info.splits['train'].num_examples,
    }

    train_set = builder.as_dataset(split=tfds.Split.TRAIN)
    val_set = train_set.take(config.data.num_val_data)

    train_set = train_set.shuffle(100000, seed=config.data.seed).repeat().map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).batch(config.data.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_set = val_set.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).batch(config.data.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return metadata, tfds.as_numpy(train_set), tfds.as_numpy(val_set)


if __name__ == '__main__':
    import omegaconf

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
                'batch_size': 1024
            },
        }
    )
    train_set, val_set = get_datasets(config)
    tfds.benchmark(train_set, num_iter=1000, batch_size=config.data.batch_size)

