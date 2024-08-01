import itertools
import pickle

import bpy
import numpy as np


def get_camera_params(camera_object: bpy.types.Object, render: bpy.types.RenderSettings):
    camera = camera_object.data

    camera_object = camera_object
    scale = render.resolution_percentage / 100
    focal_distance = camera.lens / 1000

    width, height = camera.sensor_width / 1000, camera.sensor_height / 1000
    res_x, res_y = render.resolution_x, render.resolution_y
    aspect_x, aspect_y = render.pixel_aspect_x, render.pixel_aspect_y
    m_x, m_y = None, None

    if camera.sensor_fit == 'VERTICAL' or camera.sensor_fit == 'AUTO' and aspect_x * res_x > aspect_y * res_y:
        m_y = res_y / height
        m_x = m_y * aspect_y / aspect_x
        width = res_x / m_x

    if camera.sensor_fit == 'HORIZONTAL' or camera.sensor_fit == 'AUTO' and aspect_x * res_x <= aspect_y * res_y:
        m_x = res_x / width
        m_y = m_x * aspect_x / aspect_y
        height = res_y / m_y

    return {
        'sensor_height': height,
        'sensor_width': width,
        'res_x': res_x,
        'res_y': res_y,
        'focus': focal_distance
    }


n_phi = 16
n_theta = 2

elevation = -np.pi / 6, np.pi / 4
azimuthal = 0, np.pi * 2

thetas = np.linspace(*elevation, n_theta)
phis = np.linspace(*azimuthal, n_phi, endpoint=False)
radius = 1

rotation_list = []
translation_list = []
scale_list = []

for i, (theta, phi) in enumerate(itertools.product(thetas, phis)):

    bpy.context.scene.camera.location = (
        radius * np.cos(phi) * np.cos(theta),
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(theta)
    )

    # TODO: replace with changing name explicitly
    bpy.context.scene.render.filepath = f'images/{i:04}.png'

    bpy.ops.render.render(write_still=True)

    location, rotation, scale = bpy.context.scene.camera.matrix_world.decompose()

    r = np.eye(4, dtype=np.float32)
    r[:3, :3] = rotation.to_matrix()

    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = location

    s = np.eye(4, dtype=np.float32)
    s[[0, 1, 2], [0, 1, 2]] = scale

    rotation_list.append(r)
    translation_list.append(t)
    scale_list.append(s)


data = {
    'n': n_phi * n_theta,
    'rotation': np.stack(rotation_list),
    'translation': np.stack(translation_list),
    'scale': np.stack(scale_list),
}

data.update(get_camera_params(bpy.context.scene.camera, bpy.context.scene.render))

with open('geometry.pkl', 'wb') as f:
    pickle.dump(data, f)
