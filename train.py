import os
import pathlib
import shutil
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
from utils import load, render, save


jax.config.update('jax_compilation_cache_dir', 'compilation_cache')


if __name__ == '__main__':

    config = Config.from_yaml(sys.argv[1])

    logs_folder = pathlib.Path(config.logs_folder) / config.experiment_name
    images_fine_folder = logs_folder / 'images_fine_train'
    images_coarse_folder = logs_folder / 'images_coarse_train'
    checkpoints_folder = logs_folder / 'checkpoints'

    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(images_fine_folder, exist_ok=True)
    os.makedirs(images_coarse_folder, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)

    shutil.copyfile(sys.argv[1], logs_folder / 'config.yaml')

    def loss(nerf, points, directions, pixels, keys):
        color_coarse, color_fine = nerf(points, directions, keys)
        
        loss_dict = {
            'coarse': jnp.mean((color_coarse - pixels) ** 2),
            'fine': jnp.mean((color_fine - pixels) ** 2)
        }

        return loss_dict['coarse'] + loss_dict['fine'], loss_dict

    @jax.jit
    def update(opt, nerf, points, directions, pixels, keys):
        grads, loss_dict = jax.grad(loss, has_aux=True)(nerf, points, directions, pixels, keys)
        nerf = opt.step(nerf, grads)
        return opt, nerf, loss_dict

    scene = Scene(config.scene, config.image_scale)
    key = jp.RandomKey(0)

    if config.starting_checkpoint is not None:
        opt, nerf = load(config.starting_checkpoint)
    else:
        nerf = NeRF.create(key // 1, **config.nerf, **config.nerf.mlp)

        if config.opt.use_scheduler:
            scheduler = jp.ExponentialAnnealing.create(config.opt.n_iterations, config.opt.lr, config.opt.lr_end)
        else:
            scheduler = None

        opt = jp.Adam.create(nerf, config.opt.lr, scheduler=scheduler)

    pbar = tqdm(scene.random_rays(config.opt.n_iterations - opt.t, config.opt.n_rays), initial=opt.t)

    def log(iteration):
        cam = scene.get_camera(config.train_view_to_log)
        img_coarse, img_fine = render(nerf, cam, return_coarse=True, batch_size=config.render_batch_size)

        for img, folder in zip((img_coarse, img_fine), (images_coarse_folder, images_fine_folder)):
            Image.fromarray(np.array(img)).save(folder / f'{iteration:06d}.png')

        save((opt, nerf), checkpoints_folder / f'{iteration:06d}.ckpt')

    for i, (x, d, p) in zip(range(opt.t, config.opt.n_iterations), pbar):

        if i % config.log_every == 0 or i == 0:
            log(i)

        opt, nerf, loss_dict = update(opt, nerf, x, d, p, key // config.opt.n_rays)

        pbar.set_description(f"Coarse: {loss_dict['coarse'].item():.6f}, Fine: {loss_dict['fine']:.6f}")

    log(config.opt.n_iterations)
