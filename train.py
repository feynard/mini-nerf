import os
import pathlib
import pickle
import sys

import jax
import jax.numpy as jnp
import jax_primitives as jp
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import Config
from nerf import NeRF
from scene import Scene
from utils import render


if __name__ == '__main__':

    config = Config.from_yaml(sys.argv[1])
    
    logs_folder = pathlib.Path(config.logs_folder) / config.experiment_name
    images_train_folder = logs_folder / 'images_train'
    checkpoints_folder = logs_folder / 'checkpoints'

    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(images_train_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    
    n_iterations = config.optimization.n_iterations
    
    def mse(nerf, points, directions, pixels, keys):
        color_coarse, color_fine = nerf(points, directions, keys)
        return jnp.mean((color_coarse - pixels) ** 2) + jnp.mean((color_fine - pixels) ** 2)

    @jax.jit
    def update(opt, nerf, points, directions, pixels, keys):
        loss, grads = jax.value_and_grad(mse)(nerf, points, directions, pixels, keys)
        opt, nerf = opt.step(nerf, grads)
        return opt, nerf, loss
    
    scene = Scene(config.scene, config.image_scale)
    key = jp.RandomKey(0)

    if config.starting_checkpoint is not None:
        with open(config.starting_checkpoint, 'rb') as f:
            opt, nerf = pickle.load(f)
    else:
        nerf = NeRF.create(key // 1, **config.nerf, **config.nerf.mlp)

        if config.optimization.use_scheduler:
            scheduler = jp.ExponentialAnnealing.create(n_iterations, config.optimization.lr, config.optimization.lr_end)
        else:
            scheduler = None

        opt = jp.Adam.create(nerf, config.optimization.lr, scheduler=scheduler)

    pbar = tqdm(scene.random_rays(n_iterations - opt.t, config.optimization.n_rays), initial=opt.t)
    
    def log(iteration):
        cam, img = scene.get_camera_image_pair(config.train_view_to_log)
        img_coarse, img_fine = render(nerf, cam, return_coarse=True, batch_size=config.render_batch_size)

        Image.fromarray((img_fine * 255).astype(dtype=np.uint8)).save(
            images_train_folder / f'{iteration:06d}_fine.png'
        )

        Image.fromarray((img_coarse * 255).astype(dtype=np.uint8)).save(
            images_train_folder / f'{iteration:06d}_coarse.png'
        )

        with open(checkpoints_folder / f'{iteration:06d}.ckpt', 'wb') as f:
            pickle.dump((opt, nerf), f)

    for i, (x, d, p) in zip(range(opt.t, n_iterations), pbar):

        if i % config.log_every == 0 or i == 0:
            log(i)

        opt, nerf, loss = update(opt, nerf, x, d, p, key // config.optimization.n_rays)
        
        pbar.set_description(f"Loss: {loss.item():.6f}")

    log(n_iterations)
