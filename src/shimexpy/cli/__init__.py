"""SHI core package initialization."""
from .processor import SHIProcessor
from .exceptions import SHIError
from .config import config

__all__ = ["SHIProcessor", "SHIError", "config"]
