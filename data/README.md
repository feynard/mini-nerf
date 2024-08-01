# Data Generation

Provided file `template.blend` together with `render.py` represent a basic setup for 360 scene. Just put some geometry and/or lights in the bounding sphere (of radius 1) and run
```bash
blender -b template.blend -P render.py -- --cycles-device OPTIX
```
After rendering is done, pass a folder that contains `geometry.pkl` and `images` when constructing a `Scene` class.