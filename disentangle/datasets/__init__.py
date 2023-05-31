import tensorflow as tf
from .dsprites import get_datasets
from .shapes3d import get_datasets
from .isaac3d import get_datasets
from .falcor3d import get_datasets
from .mpi3d import get_datasets
from .celeb_a import get_datasets

tf.config.experimental.set_visible_devices([], 'GPU')
