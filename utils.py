from dataclasses import dataclass
import pickle
import numpy as np


@dataclass
class Camera:

    origin: np.ndarray
    upper_left: np.ndarray
    lower_left: np.ndarray
    upper_right: np.ndarray
    res_x: int
    res_y: int


def render(nerf, camera, return_coarse: bool = False, batch_size: int = 8192):

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

    color = [
        nerf(points[i: i + batch_size], directions[i: i + batch_size], train=False, return_coarse=return_coarse)
        for i in range(0, camera.res_x * camera.res_y, batch_size)
    ]

    color = np.concatenate(color).clip(0, 1)
    
    if not return_coarse:
        return color.reshape(camera.res_y, camera.res_x, 3)
    else:
        color_coarse = color[0::2]
        color_fine = color[1::2]
        return color_coarse.reshape(camera.res_y, camera.res_x, 3), color_fine.reshape(camera.res_y, camera.res_x, 3)


def save(pytree, file):
    with open(file, 'wb') as f:
        pickle.dump(pytree, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
