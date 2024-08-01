from nerf import NeRF
from scene import Scene
from utils import render
from config import Config

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import jax_primitives as jp

from tqdm import tqdm
import sys
import os
import pathlib
import pickle


if __name__ == '__main__':

    config = Config.from_yaml(sys.argv[1])
    
    logs_folder = pathlib.Path(config.logs_folder)
    images_train_folder = logs_folder / 'images_train'
    checkpoints_folder = logs_folder / 'checkpoints'

    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(images_train_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    
    key = jp.RandomKey(0)
    nerf = NeRF.create(key // 1, **config.nerf, **config.nerf.mlp)
    opt = jp.Adam.create(nerf, config.optimization.lr)

    def mse(nerf, points, directions, pixels, keys):
        color_coarse, color_fine = nerf(points, directions, keys)
        return jnp.mean((color_coarse - pixels) ** 2) + jnp.mean((color_fine - pixels) ** 2)

    @jax.jit
    def update(opt, nerf, points, directions, pixels, keys):
        loss, grads = jax.value_and_grad(mse)(nerf, points, directions, pixels, keys)
        opt, nerf = opt.step(nerf, grads)
        return opt, nerf, loss
    
    n_iterations = config.optimization.n_iterations
    n_rays = config.optimization.n_rays

    scene = Scene(config.scene, config.image_scale)

    for i, (x, d, p) in enumerate(pbar := tqdm(scene.random_rays(n_iterations, n_rays))):

        if (i + 1) % config.log_every == 0 or i == 0:
            
            cam, img = scene.get_camera_image_pair(config.train_image_log)
            img_fine = render(nerf, cam, batch_size=8192)
            Image.fromarray(np.array(img_fine * 255, dtype=np.uint8)).save(
                images_train_folder / f'{i if i == 0 else i + 1:06d}.png'
            )

            with open(checkpoints_folder / f'{i if i == 0 else i + 1:06d}.ckpt', 'wb') as f:
                pickle.dump(nerf, f)

        opt, nerf, loss = update(opt, nerf, jnp.array(x), jnp.array(d), jnp.array(p), key // n_rays)
        
        pbar.set_description(f"Loss: {loss.item():.6f}")
