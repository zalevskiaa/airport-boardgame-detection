import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parent.parent
CAMERA_SOURCE = 'http://192.168.1.6:8080/video'


def add_project_root():
    sys.path.append(str(PROJECT_ROOT))


add_project_root()
# print('\n'.join(sys.path))
