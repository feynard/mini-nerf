import os
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
    ):

        with open(os.path.join(main_folder, 'geometry.pkl'), 'rb') as f:
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

        #self.o, self.ul, self.ll, self.ur = (self.cone @ s.mT @ r.mT @ t.mT).permute(1, 0, 2) [:, :, :3]
        self.o, self.ul, self.ll, self.ur = (t @ r @ s @ cone.T).transpose(2, 0, 1) [:, :, :3]

        #self.rotation, self.translation, self.scale = r, t, s

        images_folder = os.path.join(main_folder, 'images')
        images = [
            np.array(Image.open(os.path.join(images_folder, f'{i:04}.png')).convert('RGB').resize((384, 512)), dtype=np.float32) / 255
            for i in range(self.n_images)
        ]

        self.images = np.stack(images)

        self.res_x, self.res_y = data['res_x'], data['res_y']
        self.res_x, self.res_y = 384, 512

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
