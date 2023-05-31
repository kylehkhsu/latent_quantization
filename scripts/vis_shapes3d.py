import h5py
import ipdb
import numpy as np
import imageio
import pathlib
import einops

path = '/iris/u/kylehsu/data/downloads/3d-shapes_3dshapesCg9u2Yuv8nalDzoIGnQ014jaY8sTWpgYmypbV2m-F4U.h5'
path = pathlib.Path(path)
images_all = h5py.File(path, 'r')['images']

source_sizes = [10, 10, 10, 8, 4, 15]

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
    imageio.imwrite(f'vis/paper/shapes3d_random.png', images)
