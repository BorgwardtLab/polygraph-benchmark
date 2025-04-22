import warnings
from importlib.metadata import version

from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

__version__ = version("polygraph")
