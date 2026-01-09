"""Custom exceptions for the SHI package."""

class SHIError(Exception):
    """Base exception for SHI errors."""
    pass

class ImageNotFoundError(SHIError):
    """Raised when required image files are not found."""
    pass

class ProcessingError(SHIError):
    """Raised when there's an error during image processing."""
    pass

class ConfigurationError(SHIError):
    """Raised when there's an error in configuration."""
    pass

class CleanupError(SHIError):
    """Raised when there's an error during cleanup operations."""
    pass
