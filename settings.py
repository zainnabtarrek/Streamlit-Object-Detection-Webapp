from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'


SOURCES_LIST = [IMAGE]

DEFAULT_IMAGE = 'office_4.jpg'
DEFAULT_DETECT_IMAGE = 'office_4_detected.jpg'



# ML Model config

DETECTION_MODEL = 'yolov8n.pt'


SEGMENTATION_MODEL = 'yolov8n-seg.pt'

