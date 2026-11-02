"""Mesh images with mesh-screenshot. Needs a display."""
import json
import subprocess
from pathlib import Path

SCREENSHOT = Path(__file__).resolve().parents[2] / 'build/bin/mesh-screenshot'
WINDOW_WIDTH, WINDOW_HEIGHT = 1506, 907


def camera(center, view):
    """The viewers' Copy Camera JSON, from the view center and the view matrix."""
    return json.dumps({'farClip': 20.0, 'fov': 45.0, 'frontDir': 'Z Forward', 'navigateStyle': 'Turntable',
                       'nearClip': 0.00499999988824129, 'projectionMode': 'Perspective', 'upDir': 'Y Up',
                       'viewCenter': center, 'viewMat': view, 'viewRelativeMode': 'Center Relative',
                       'windowHeight': WINDOW_HEIGHT, 'windowWidth': WINDOW_WIDTH})


def screenshot(source, output, options, target_length=None):
    """source: a mesh, or a result folder of remesher-cli, which brings its own target length."""
    command = [str(SCREENSHOT), str(source), str(output), '--no-ground', '--ssaa', '2', *options]
    if target_length is not None:
        command += ['--target-length', repr(float(target_length))]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
