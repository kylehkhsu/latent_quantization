import ipdb
import numpy as np
import imageio
import pathlib
import einops

path = '/iris/u/kylehsu/data/mpi3d_raw/images.npy'
path = pathlib.Path(path)
images_all = np.load(path)
source_sizes = [4, 4, 2, 3, 3, 40, 40]
source_names = ['object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'horizontal_axis', 'vertical_axis']
source_bases = np.array(np.prod(source_sizes) / np.cumprod(source_sizes), np.int32)

pathlib.Path(f'vis/mpi3d').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'vis/paper/').mkdir(parents=True, exist_ok=True)

if False:   # sources
    for source_index, source_size in enumerate(source_sizes):
        indices = source_bases[source_index] * np.arange(source_size)
        print(indices)
        images = []
        for i in indices:
            images.append(images_all[i])
        images = np.stack(images, axis=0)
        images = einops.rearrange(images, 'n h w c -> h (n w) c')
        imageio.imwrite(f'vis/mpi3d/{source_names[source_index]}.png', images)

if True:    # random
    np.random.seed(42)
    width = 6
    height = 2
    indices = np.random.choice(np.prod(source_sizes), width * height, replace=False)
    images = []
    for i in indices:
        images.append(images_all[i])
    images = np.stack(images, axis=0)
    images = einops.rearrange(images, '(width height) h w c -> (height h) (width w) c', width=width, height=height)
    imageio.imwrite(f'vis/paper/mpi3d_random.png', images)

