from typing import Dict, List, Optional, Self, Tuple

import jax
import jax.numpy as jnp
import jax_primitives as jp
from jax_primitives import Dynamic, modelclass


@modelclass
class MLP:

    main_layers: Dynamic[List[jp.Linear]]
    sigma_layer: Dynamic[jp.Linear]
    color_layers: Dynamic[Dict[str, jp.Linear | jax.Array]]
    conditioned_layers: Tuple[int, ...]

    def __init__(
        self,
        key: jax.Array,
        x_dim: int,
        d_dim: int,
        inner_dim: int,
        n_layers: int,
        conditioned_layers: Tuple[int, ...],
        *args,
        **kwargs
    ) -> Self:

        keys = jax.random.split(key, n_layers + 1)

        self.main_layers = [jp.Linear(x_dim, inner_dim, keys[0])]

        for i in range(1, n_layers + 1):
            if i in conditioned_layers:
                self.main_layers.append(jp.Linear(inner_dim + x_dim, inner_dim, keys[i]))
            else:
                self.main_layers.append(jp.Linear(inner_dim, inner_dim, keys[i]))

        self.sigma_layer = jp.Linear(inner_dim, 1, keys[n_layers])

        keys = jax.random.split(key, 3)

        self.color_layers = {
            'backbone': jp.Linear(inner_dim, inner_dim // 2, keys[0], bias=False),
            'direction': jp.Linear(d_dim, inner_dim // 2, keys[1], bias=False),
            'bias': jnp.zeros(inner_dim // 2),
            'final': jp.Linear(inner_dim // 2, 3, keys[2])
        }

        self.conditioned_layers = conditioned_layers


    def __call__(
        self,
        x: jax.Array,
        d: jax.Array,
        add_noise: bool = True,
        sigma_noise_key: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        
        y = x

        for i in range(len(self.main_layers)):
            if i in self.conditioned_layers:
                y = jnp.concatenate((y, x), -1)

            y = self.main_layers[i](y)

            if i != len(self.main_layers) - 1:
                y = jax.nn.relu(y)

        sigma = self.sigma_layer(y)

        if add_noise:
            sigma = sigma + jax.random.normal(sigma_noise_key, sigma.shape)

        sigma = jax.nn.relu(sigma)
        sigma = sigma.squeeze(-1)

        color = self.color_layers['backbone'](y) + self.color_layers['direction'](d) + self.color_layers['bias']
        color = jax.nn.relu(color)
        color = self.color_layers['final'](color)
        color = jax.nn.sigmoid(color)

        return sigma, color


def sinusoidal_emb(x, d):
    # TODO: try other implementations, e.g. from here 
    # https://stackoverflow.com/questions/5347065/interleaving-two-numpy-arrays-efficiently

    f = 2 ** jnp.arange(d) * jnp.pi

    y = jnp.empty(x.shape + (2 * d, ))
    y = y.at[..., 0::2].set(jnp.sin(x[..., jnp.newaxis] * f))
    y = y.at[..., 1::2].set(jnp.cos(x[..., jnp.newaxis] * f))
    y = y.reshape(*y.shape[:-2], -1)
    
    return y


@modelclass
class NeRF:

    net_coarse: Dynamic[MLP]
    net_fine: Dynamic[MLP]
    sampling_depth: float
    n_coarse_samples: int
    n_fine_samples: int
    x_pos_dim: int
    d_pos_dim: int


    def __init__(
        self,
        key: jax.Array,
        x_pos_dim: int = 10,
        d_pos_dim: int = 4,
        inner_dim: int = 256,
        n_layers: int = 6,
        conditioned_layers: Tuple[int, ...] = (4, ),
        sampling_depth: float = 1.0,
        n_coarse_samples: int = 64,
        n_fine_samples: int = 128,
        *args,
        **kwargs
    ) -> Self:
        
        key_coarse, key_fine = jax.random.split(key, 2)

        self.net_coarse = MLP(key_coarse, x_pos_dim * 6, d_pos_dim * 6, inner_dim, n_layers, conditioned_layers)
        self.net_fine = MLP(key_fine, x_pos_dim * 6, d_pos_dim * 6, inner_dim, n_layers, conditioned_layers)

        self.sampling_depth = sampling_depth
        self.n_coarse_samples = n_coarse_samples
        self.n_fine_samples = n_fine_samples
        self.x_pos_dim = x_pos_dim
        self.d_pos_dim = d_pos_dim

    def ray_march(self, color, sigma, samples):

        delta = samples[1: ] - samples[: -1]
        delta = jnp.concatenate((delta, self.sampling_depth - samples[-1:]))
        
        t = jnp.exp(-jnp.cumsum(sigma * delta))
        w = t * (1 - jnp.exp(-sigma * delta))

        return jnp.sum(color * jnp.expand_dims(w, -1), 0), w / (w.sum() + 1e-7)

    def __call__(
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
            sigma_coarse, color_coarse = self.net_coarse(x, d, add_noise=True, sigma_noise_key=coarse_net_key)
        else:
            sigma_coarse, color_coarse = self.net_coarse(x, d, add_noise=False)

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
            sigma_fine, color_fine = self.net_fine(x, d, add_noise=True, sigma_noise_key=fine_net_key)
        else:
            sigma_fine, color_fine = self.net_fine(x, d, add_noise=False)

        color_fine, _ = self.ray_march(color_fine, sigma_fine, samples_final)

        if train or return_coarse:
            return color_coarse, color_fine
        else:
            return color_fine
        