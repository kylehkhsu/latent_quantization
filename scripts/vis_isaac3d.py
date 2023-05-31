import ipdb
import numpy as np
import imageio
import pathlib
import einops

path = '/iris/u/kylehsu/data/extracted/ZIP.ucid_1OmQ1G2wnm6eTsSFGTKFZZAh5D3nQTW1BJJAzFAGx0vln9Mn-i6CBfxnWXt9uUrHtsCD9pQkz128/Isaac3D_down128'
path = pathlib.Path(path)
labels = np.load(path / 'labels.npy')
source_sizes = [3, 8, 5, 4, 4, 4, 6, 4, 4]
source_names = ['object_shape', 'robot_x-move', 'robot_y-move', 'camera_height', 'object_scale',
                'lighting_intensity', 'lighting_y-dir', 'object_color', 'wall_color']
source_bases = np.array(np.prod(source_sizes) / np.cumprod(source_sizes), np.int32)

pathlib.Path(f'vis/isaac3d').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'vis/paper/').mkdir(parents=True, exist_ok=True)

if True:    # paper figure 1
    for i in range(3):
        for j in range(0, 5, 2):
            indices = np.array([i, j, 2, 0, 3, 0, 2, 1, 1])
            index = np.ravel_multi_index(indices, source_sizes)
            image_path = path / 'images' / f'{index:06}.png'
            image = imageio.imread_v2(image_path)
            imageio.imwrite(f'vis/paper/isaac3d_i={i}_j={j}.png', image)

if False:   # sources
    for source_index, source_size in enumerate(source_sizes):
        indices = source_bases[source_index] * np.arange(source_size)
        print(indices)
        ipdb.set_trace()
        images = []
        for i in indices:
            image_path = path / 'images' / f'{i:06}.png'
            image = imageio.imread_v2(image_path)
            images.append(image)
        images = np.stack(images, axis=0)
        images = einops.rearrange(images, 'n h w c -> h (n w) c')
        imageio.imwrite(f'vis/isaac3d/{source_names[source_index]}.png', images)

if False:   # random
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
    imageio.imwrite(f'vis/paper/isaac3d_random.png', images)

