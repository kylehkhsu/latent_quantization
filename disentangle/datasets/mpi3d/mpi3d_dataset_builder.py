"""mpi3d dataset."""

import tensorflow_datasets as tfds
import numpy as np
import ipdb


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mpi3d dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(64, 64, 3)),
                'object_color': tfds.features.ClassLabel(num_classes=4),
                'object_shape': tfds.features.ClassLabel(num_classes=4),
                'object_size': tfds.features.ClassLabel(num_classes=2),
                'camera_height': tfds.features.ClassLabel(num_classes=3),
                'background_color': tfds.features.ClassLabel(num_classes=3),
                'horizontal_axis': tfds.features.ClassLabel(num_classes=40),
                'vertical_axis': tfds.features.ClassLabel(num_classes=40),
            }),
            supervised_keys=None,  # Set to `None` to disable
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download('https://drive.google.com/uc?id=1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm')
        return {
            'train': self._generate_examples(path)
        }

    def _generate_examples(self, path):
        """Yields examples."""

        images = np.load(path)['images']
        label_sizes = [4, 4, 2, 3, 3, 40, 40]

        for i, image in enumerate(images):
            coordinates = np.unravel_index(i, label_sizes)
            record = {
                'image': image,
                'object_color': coordinates[0],
                'object_shape': coordinates[1],
                'object_size': coordinates[2],
                'camera_height': coordinates[3],
                'background_color': coordinates[4],
                'horizontal_axis': coordinates[5],
                'vertical_axis': coordinates[6],
            }
            yield i, record
