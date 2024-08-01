# Data Generation

Provided file `template.blend` together with `render.py` represent a basic setup for 360 scene. Just put some geometry and lights in the bounding sphere (of radius 1) and run (OPTIX is for NVIDIA cards, look [here](https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html#command-line-args-cycles-render-options) for the rest)
```bash
blender -b template.blend -P render.py -- --cycles-device OPTIX
```
After rendering is done, pass a folder that contains `geometry.pkl` and `images` when constructing a `Scene` class.

_Note: it seems there can be a bug in how Blender runs Python when using a virtual environment, ensure you don't run the command under any virtual environment_