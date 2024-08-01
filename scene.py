import pathlib
import pickle
from typing import Generator, Tuple

import numpy as np
from PIL import Image

from utils import Camera


class Scene:
    """
    for points, directions, pixels in scene.random_rays(1000):
        ...

    for points, directions, pixels in scene.image(23):
        ...
    """
    
    def __init__(
        self,
        main_folder: str,
        image_scale: float = 1.0
    ):
        
        main_folder = pathlib.Path(main_folder)

        with open(main_folder / 'geometry.pkl', 'rb') as f:
            data = pickle.load(f)

        self.n_images = data['n']

        w, h = data['sensor_width'], data['sensor_height']
        f = data['focus']

        r = np.array(data['rotation'], dtype=np.float32)
        t = np.array(data['translation'], dtype=np.float32)
        s = np.array(data['scale'], dtype=np.float32)

        cone = np.array([
            [      0,      0,     0,      1], # origin
            [ -w / 2,  h / 2,    -f,      1], # upper left
            [ -w / 2, -h / 2,    -f,      1], # lower left
            [  w / 2,  h / 2,    -f,      1]  # upper right
        ], dtype=np.float32)

        self.o, self.ul, self.ll, self.ur = (t @ r @ s @ cone.T).transpose(2, 0, 1) [:, :, :3]

        images_folder = main_folder / 'images'
        images = [
            Image.open(images_folder / f'{i:04}.png').convert('RGB')
            for i in range(self.n_images)
        ]

        if image_scale != 1.0:
            for i, img in enumerate(images):
                images[i] = img.resize((round(r * image_scale) for r in img.size))

        self.images = np.stack([np.array(img, dtype=np.float32) / 255 for img in images])

        self.res_y, self.res_x = self.images.shape[1: 3]

        self.u = (self.ur - self.ul) / self.res_x
        self.v = (self.ll - self.ul) / self.res_y

    def random_rays(self, n_iterations: int, n_sample_rays: int = 128) -> Generator:

        generator = np.random.default_rng()

        for _ in range(n_iterations):

            n, i, j = np.unravel_index(
                generator.choice(
                    a=self.n_images * self.res_y * self.res_x,
                    size=n_sample_rays,
                    replace=self.n_images * self.res_y * self.res_x < n_sample_rays
                ),
                (self.n_images, self.res_y, self.res_x)
            )

            points = self.ul[n] + self.v[n] * np.expand_dims(i, -1) + self.u[n] * np.expand_dims(j, -1)

            directions = points - self.o[n]
            directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

            pixels = self.images[n, i, j]

            yield points, directions, pixels

    def get_camera_image_pair(self, i: int) -> Tuple[Camera, np.ndarray]:
        cam = Camera(
            origin=self.o[i],
            upper_left=self.ul[i],
            lower_left=self.ll[i],
            upper_right=self.ur[i],
            res_x=self.res_x,
            res_y=self.res_y
        )

        img = self.images[i]

        return cam, img
