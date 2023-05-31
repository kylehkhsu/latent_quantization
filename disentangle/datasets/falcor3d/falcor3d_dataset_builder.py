"""falcor3d dataset."""

import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for falcor3d dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(128, 128, 3)),
            'label': tfds.features.Tensor(shape=(7,), dtype=np.float64)
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('https://drive.google.com/uc?id=1XAQfFK1x6cpN1eiovbP0hVfLTm5SsSoJ')
    return {
        'train': self._generate_examples(path)
    }

  def _generate_examples(self, path):
    """Yields examples."""

    labels = np.load(path / 'Falcor3D_down128/train-rec.labels')
    for i, label in enumerate(labels):
        image_path = path / 'Falcor3D_down128' / 'images' / f'{i:06}.png'
        yield i, {
            'image': image_path,
            'label': label
        }
