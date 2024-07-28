from typing import Tuple, Self, Optional, List

from dataclasses import dataclass
from functools import partial

from config import Config

import jax_primitives as jp

import jax.numpy as jnp
import jax


@jp.modelclass
class MLP:

    main_layers: jp.Dynamic[List[jp.Linear]]
    sigma_layer: jp.Dynamic[jp.Linear]
    color_backbone: jp.Dynamic[List[jp.Linear]]
    color_bias: jp.Dynamic[jax.Array]
    color_final_layer: jp.Dynamic[jp.Linear]
    conditioned_layers: Tuple[int, ...]

    @classmethod
    def create(
        cls,
        key: jax.Array,
        x_dim: int,
        d_dim: int,
        inner_dim: int,
        n_layers: int,
        conditioned_layers: Tuple[int, ...]
    ) -> Self:

        keys = jax.random.split(key, n_layers + 1)

        main_layers = [jp.Linear.create(x_dim, inner_dim, keys[0])]

        for i in range(1, n_layers + 1):
            if i == 4:
                main_layers.append(jp.Linear.create(inner_dim + x_dim, inner_dim, keys[i]))
            else:
                main_layers.append(jp.Linear.create(inner_dim, inner_dim, keys[i]))

        sigma_layer = jp.Linear.create(inner_dim, 1, keys[n_layers])

        keys = jax.random.split(key, 3)

        color_backbone = [
            jp.Linear.create(inner_dim, inner_dim // 2, keys[0]),
            jp.Linear.create(d_dim, inner_dim // 2, keys[1])
        ]
        color_bias = jnp.zeros(inner_dim // 2)
        color_final_layer = jp.Linear.create(inner_dim // 2, 3, keys[2])

        return cls(main_layers, sigma_layer, color_backbone, color_bias, color_final_layer, conditioned_layers)

    @partial(jax.jit, static_argnames='train')
    def __call__(
        self,
        x: jax.Array,
        d: jax.Array,
        sigma_noise_key: Optional[jax.Array] = None,
        train: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
        
        y = x

        for i in range(len(self.main_layers)):
            if i in self.conditioned_layers:
                y = jnp.concatenate((y, x), -1)

            y = self.main_layers[i](y)

            if i != len(self.main_layers) - 1:
                y = jax.nn.relu(y)

        sigma = self.sigma_layer(y)

        if train:
            sigma = sigma + jax.random.normal(sigma_noise_key, sigma.shape)

        sigma = jax.nn.relu(sigma)
        sigma = sigma.squeeze(-1)

        color = self.color_backbone[0](y) + self.color_backbone[1](d) + self.color_bias
        color = jax.nn.relu(color)
        color = self.color_final_layer(color)
        color = jax.nn.sigmoid(color)

        return sigma, color


@partial(jax.jit, static_argnames='d')
def sinusoidal_emb(x, d):
    # TODO: try other implementations, e.g. from here 
    # https://stackoverflow.com/questions/5347065/interleaving-two-numpy-arrays-efficiently

    f = 2 ** jnp.arange(d) * jnp.pi

    y = jnp.empty(x.shape + (2 * d, ))
    y = y.at[..., 0::2].set(jnp.sin(x[..., jnp.newaxis] * f))
    y = y.at[..., 1::2].set(jnp.cos(x[..., jnp.newaxis] * f))
    y = y.reshape(*y.shape[:-2], -1)
    
    return y


@jp.modelclass
class NeRF:

    net_coarse: jp.Dynamic[MLP]
    net_fine: jp.Dynamic[MLP]
    sampling_depth: float
    n_coarse_samples: int
    n_fine_samples: int
    x_pos_dim: int
    d_pos_dim: int


    @classmethod
    def create(
        cls,
        key: jax.Array,
        x_pos_dim: int = 10,
        d_pos_dim: int = 4,
        inner_dim: int = 256,
        n_layers: int = 6,
        conditioned_layers: Tuple[int, ...] = (4, ),
        sampling_depth: float = 1.0,
        n_coarse_samples: int = 64,
        n_fine_samples: int = 128,
    ) -> Self:
        
        key_coarse, key_fine = jax.random.split(key, 2)

        net_coarse = MLP.create(key_coarse, x_pos_dim * 6, d_pos_dim * 6, inner_dim, n_layers, conditioned_layers)
        net_fine = MLP.create(key_fine, x_pos_dim * 6, d_pos_dim * 6, inner_dim, n_layers, conditioned_layers)
        
        return cls(net_coarse, net_fine, sampling_depth, n_coarse_samples, n_fine_samples, x_pos_dim, d_pos_dim)


    def ray_march(self, color, sigma, samples):

        delta = samples[1: ] - samples[: -1]
        delta = jnp.concatenate((delta, self.sampling_depth - samples[-1:]))
        
        t = jnp.exp(-jnp.cumsum(sigma * delta))
        w = t * (1 - jnp.exp(-sigma * delta))

        return jnp.sum(color * jnp.expand_dims(w, -1), 0), w / (w.sum() + 1e-7)


    @partial(jax.jit, static_argnames=('train', 'return_coarse'))
    def __call__(
        self,
        points: jax.Array,
        directions: jax.Array,
        keys: Optional[jax.Array] = None,
        train: bool = True,
        return_coarse: bool = False
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:

        return self.render(points, directions, keys, train, return_coarse)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None))
    def render(
        self,
        point: jax.Array,
        direction: jax.Array,
        key: Optional[jax.Array] = None,
        train: bool = True,
        return_coarse: bool = False
    ) -> jax.Array | Tuple[jax.Array, jax.Array]:

        segment_size = self.sampling_depth / self.n_coarse_samples

        if train:
            key, coarse_sampling_key = jax.random.split(key, 2)

            samples_coarse = jnp.linspace(0, self.sampling_depth, self.n_coarse_samples, endpoint=False)

            samples_coarse = \
                samples_coarse + \
                segment_size * jax.random.uniform(coarse_sampling_key, (self.n_coarse_samples,))
        else:
            samples_coarse = jnp.linspace(0, self.sampling_depth, self.n_coarse_samples)
        
        x = point + direction * jnp.expand_dims(samples_coarse, -1)
        d = -direction

        x = sinusoidal_emb(x, self.x_pos_dim)
        d = sinusoidal_emb(d, self.d_pos_dim)

        if train:
            key, coarse_net_key = jax.random.split(key, 2)
            sigma_coarse, color_coarse = self.net_coarse(x, d, coarse_net_key)
        else:
            sigma_coarse, color_coarse = self.net_coarse(x, d, train=False)

        color_coarse, w = self.ray_march(color_coarse, sigma_coarse, samples_coarse)

        # Inverse sampling
        if train:
            key, inverse_sampling_key = jax.random.split(key, 2)
            
            samples_fine = jnp.interp(
                jax.random.uniform(inverse_sampling_key, (self.n_fine_samples, )),
                jnp.cumsum(w),
                samples_coarse
            )
        else:
            samples_fine = jnp.interp(
                jnp.linspace(0, 1, self.n_fine_samples),
                jnp.cumsum(w),
                samples_coarse
            )
        
        samples_final = jnp.sort(jnp.concatenate((samples_coarse, samples_fine)))

        x = point + direction * jnp.expand_dims(samples_final, -1)
        x = sinusoidal_emb(x, self.x_pos_dim)

        if train:
            key, fine_net_key = jax.random.split(key, 2)
            sigma_fine, color_fine = self.net_fine(x, d, fine_net_key)
        else:
            sigma_fine, color_fine = self.net_fine(x, d, train=False)

        color_fine, _ = self.ray_march(color_fine, sigma_fine, samples_final)

        if train or return_coarse:
            return color_coarse, color_fine
        else:
            return color_fine
        