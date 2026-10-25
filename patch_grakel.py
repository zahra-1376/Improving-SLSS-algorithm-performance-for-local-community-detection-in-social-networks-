"""
Patch for grakel library compatibility with newer numpy versions.

This module fixes the import issue with ComplexWarning in numpy 2.x
where ComplexWarning has been moved from numpy to numpy.exceptions
"""

import sys

# Define ComplexWarning for backward compatibility with grakel
try:
    from numpy.exceptions import ComplexWarning
except ImportError:
    # For older numpy versions where it's still in numpy
    from numpy import ComplexWarning


def patch_grakel_import():
    """
    Patch the numpy import to make ComplexWarning available where grakel expects it.
    """
    # Add ComplexWarning to numpy's main module if it's not there
    import numpy as np
    
    if not hasattr(np, 'ComplexWarning'):
        np.ComplexWarning = ComplexWarning
    
    # Also add to numpy.exceptions if that's where it should be
    try:
        import numpy.exceptions
        if not hasattr(numpy.exceptions, 'ComplexWarning'):
            numpy.exceptions.ComplexWarning = ComplexWarning
    except ImportError:
        # If numpy.exceptions doesn't exist in this version, we don't need to worry about it
        pass


# Apply the patch immediately when this module is imported
patch_grakel_import()