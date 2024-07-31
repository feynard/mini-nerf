from nerf import NeRF
from scene import Scene
from utils import render

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import jax_primitives as jp

from tqdm import tqdm


if __name__ == '__main__':
    
    key = jp.RandomKey(0)
    nerf = NeRF.create(key // 1)
    opt = jp.Adam.create(nerf, 0.0001)

    def mse(nerf, points, directions, pixels, keys):
        color_coarse, color_fine = nerf(points, directions, keys)
        return jnp.mean((color_coarse - pixels) ** 2) + jnp.mean((color_fine - pixels) ** 2)

    @jax.jit
    def update(opt, nerf, points, directions, pixels, keys):
        loss, grads = jax.value_and_grad(mse)(nerf, points, directions, pixels, keys)
        opt, nerf = opt.step(nerf, grads)
        return opt, nerf, loss
    
    n_iterations = 100_000
    n_rays = 4096
    scene = Scene('roza')

    i = 0
    for points, directions, pixels in (pbar := tqdm(scene.random_rays(n_iterations, n_rays))):
        
        opt, nerf, loss = update(opt, nerf, jnp.array(points), jnp.array(directions), jnp.array(pixels), key // n_rays)
        pbar.set_description(f"Loss: {loss.item():.6f}")

        if i % 1_000 == 0:
            cam, img = scene.get_camera_image_pair(13)
            img_coarse, img_fine = render(nerf, cam, return_coarse=True, batch_size=8192)
            Image.fromarray(np.array(img_coarse * 255, dtype=np.uint8)).save(f'render_coarse_{i}.png')
            Image.fromarray(np.array(img_fine * 255, dtype=np.uint8)).save(f'render_fine_{i}.png')

        i += 1

