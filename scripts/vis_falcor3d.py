import ipdb
import numpy as np
import imageio
import pathlib
import einops

path = '/iris/u/kylehsu/data/extracted/ZIP.ucid_1XAQfFK1x6cpN1eiovbP0hVfLTm5SsSoJhDRauwMTtchNJCfP_eQlI5w2qRzRNUOLtzjBJJMCa2E/Falcor3D_down128'
path = pathlib.Path(path)
labels = np.load(path / 'train-rec.labels')
source_sizes = [5, 6, 6, 6, 6, 6, 6]
source_names = ['lighting_intensity', 'lighting_x-dir', 'lighting_y-dir', 'lighting_z-dir',
                        'camera_x-pos', 'camera_y-pos', 'camera_z-pos']
source_bases = np.array(np.prod(source_sizes) / np.cumprod(source_sizes), np.int32)

pathlib.Path(f'vis/falcor3d').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'vis/paper').mkdir(parents=True, exist_ok=True)

if False:   # sources
    for source_index, source_size in enumerate(source_sizes):
        indices = source_bases[source_index] * np.arange(source_size)
        print(indices)
        images = []
        for i in indices:
            image_path = path / 'images' / f'{i:06}.png'
            image = imageio.imread_v2(image_path)
            images.append(image)
        images = np.stack(images, axis=0)
        images = einops.rearrange(images, 'n h w c -> h (n w) c')
        imageio.imwrite(f'vis/falcor3d/{source_names[source_index]}.png', images)

if True:    # random
    np.random.seed(42)
    width = 6
    height = 2
    indices = np.random.choice(np.prod(source_sizes), width * height, replace=False)
    images = []
    for i in indices:
        image_path = path / 'images' / f'{i:06}.png'
        image = imageio.imread_v2(image_path)
        images.append(image)
    images = np.stack(images, axis=0)
    images = einops.rearrange(images, '(width height) h w c -> (height h) (width w) c', width=width, height=height)
    imageio.imwrite(f'vis/paper/falcor3d_random.png', images)



