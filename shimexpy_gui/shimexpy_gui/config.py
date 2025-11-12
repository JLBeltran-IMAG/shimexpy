"""
Configuration for the ShimExPy GUI.

This module contains configuration definitions for the GUI.
"""

# Main window configuration
WINDOW_CONFIG = {
    'title': 'ShimExPy',
    'width': 1200,
    'height': 800,
    'min_width': 800,
    'min_height': 600,
}

# Processing configuration
PROCESSING_CONFIG = {
    'default_contrast_type': 'amplitude',
    'default_filter': 'none',
}

# Application messages
MESSAGES = {
    'welcome': 'Welcome to ShimExPy',
    'no_image': 'No image loaded',
    'processing_done': 'Processing completed',
    'processing_error': 'Error during processing',
    'save_success': 'Image saved successfully',
    'save_error': 'Error saving the image',
}

# Supported image formats
IMAGE_FORMATS = [
    'tiff', 'tif', 'png', 'jpg', 'jpeg', 'bmp'
]
