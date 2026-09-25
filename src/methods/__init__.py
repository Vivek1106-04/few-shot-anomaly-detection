"""S3/S4 — the detection methods, behind one interface.

`from methods import build_method` is the only way the experiment runner obtains a
method, so adding a method cannot change anything outside its own file.
"""

from .base import AnomalyMethod, build_method, known_methods

__all__ = ["AnomalyMethod", "build_method", "known_methods"]
