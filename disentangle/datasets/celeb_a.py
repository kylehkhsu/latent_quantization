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
    x = data['image']   # (218, 178, 3)
    x = tf.image.resize(x, [64, 64], method='bicubic', antialias=True)
    x = tf.transpose(x, [2, 0, 1])   # (3, 218, 178)
    x = x / 255.
    x = tf.clip_by_value(x, 0., 1.)
    ret['x'] = x

    # boolean
    attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    # pixel index
    landmarks = ['lefteye_x', 'lefteye_y', 'leftmouth_x', 'leftmouth_y', 'nose_x', 'nose_y', 'righteye_x', 'righteye_y', 'rightmouth_x', 'rightmouth_y']
    ret['z'] = tf.stack(
        [int(data['attributes'][attribute]) for attribute in attributes] +
        [data['landmarks'][landmark] for landmark in landmarks]
    )
    return ret


def get_datasets(config):
    possible_dirs = config.data.possible_dirs
    while len(possible_dirs) > 0:
        possible_dir = pathlib.Path(possible_dirs.pop(0))
        try:
            builder = tfds.builder('celeb_a', data_dir=possible_dir)
            builder.download_and_prepare()
            break
        except PermissionError as e:
            print(e)

    train_set = builder.as_dataset(split=tfds.Split.TRAIN)
    val_set = builder.as_dataset(split=tfds.Split.VALIDATION)

    train_set = train_set.shuffle(100000, seed=config.data.seed).repeat().map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).batch(config.optim.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_set = val_set.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE).batch(config.optim.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return tfds.as_numpy(train_set), tfds.as_numpy(val_set)


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
                'num_val_data': 10000
            },
            'optim': {
                'batch_size': 1024
            }
        }
    )
    train_set, val_set = get_datasets(config)
    tfds.benchmark(train_set, num_iter=1000, batch_size=config.optim.batch_size)

