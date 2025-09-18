from importlib.metadata import version
from typing import TypeVar

__version__ = version("polygraph")

GraphType = TypeVar("GraphType", contravariant=True)
