# Make this directory a Python package
try:
    from . import checker_bindings
except ImportError:
    import warnings
    warnings.warn("checker_bindings module not found. Please build the C++ extension first with 'make all' or 'pip install -e .'") 