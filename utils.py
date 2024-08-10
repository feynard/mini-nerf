import pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax_primitives import Dynamic, pytree


@pytree
class Camera:

    origin: Dynamic[jax.Array]
    upper_left: Dynamic[jax.Array]
    lower_left: Dynamic[jax.Array]
    upper_right: Dynamic[jax.Array]
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

    points = jnp.linspace(
        jnp.linspace(camera.upper_left, camera.upper_right, camera.res_x),
        jnp.linspace(camera.lower_left, camera.lower_left + camera.upper_right - camera.upper_left, camera.res_x),
        camera.res_y
    )

    points = points.reshape(-1, 3)

    directions = points - camera.origin
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)

    batch_render = jax.jit(partial(nerf.__call__, train=False, return_coarse=return_coarse))

    color_batches = [
        batch_render(points[i: i + batch_size], directions[i: i + batch_size])
        for i in range(0, camera.res_x * camera.res_y, batch_size)
    ]

    image_shape = camera.res_y, camera.res_x, 3

    if return_coarse:
        color_coarse, color_fine = [
            jnp.concatenate([c[i] for c in color_batches]).clip(0, 1).reshape(*image_shape)
            for i in range(2)
        ]

        return [jnp.array(c * 255).astype(dtype=jnp.uint8) for c in [color_coarse, color_fine]]
    else:
        color_fine = jnp.concatenate(color_batches).clip(0, 1).reshape(*image_shape)
        return (color_fine * 255).astype(jnp.uint8)


def save(pytree, file):
    with open(file, 'wb') as f:
        pickle.dump(pytree, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
