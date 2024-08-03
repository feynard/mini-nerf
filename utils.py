import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:

    origin: np.ndarray
    upper_left: np.ndarray
    lower_left: np.ndarray
    upper_right: np.ndarray
    res_x: int
    res_y: int


def render(nerf, camera, return_coarse: bool = False, batch_size: int = 1024):

    '''
    Train:
    cam, img_true = scene.get_camera_image_pair(i)
    img_pred = render(nerf, cam)
    
    Validation:
    cam = generate_view(...)
    img = render(nerf, cam)
    '''

    points = np.linspace(
        np.linspace(camera.upper_left, camera.upper_right, camera.res_x),
        np.linspace(camera.lower_left, camera.lower_left + camera.upper_right - camera.upper_left, camera.res_x),
        camera.res_y
    )

    points = points.reshape(-1, 3)

    directions = points - camera.origin
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    color_batches = [
        nerf(points[i: i + batch_size], directions[i: i + batch_size], train=False, return_coarse=return_coarse)
        for i in range(0, camera.res_x * camera.res_y, batch_size)
    ]

    if return_coarse:
        color_coarse = np.concatenate([c[0] for c in color_batches]).clip(0, 1).reshape(camera.res_y, camera.res_x, 3)
        color_fine = np.concatenate([c[1] for c in color_batches]).clip(0, 1).reshape(camera.res_y, camera.res_x, 3)
        return color_coarse, color_fine
    else:
        return np.concatenate(color_batches).clip(0, 1).reshape(camera.res_y, camera.res_x, 3)


def save(pytree, file):
    with open(file, 'wb') as f:
        pickle.dump(pytree, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
