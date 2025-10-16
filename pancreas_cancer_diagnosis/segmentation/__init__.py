"""
Segmentation Module
===================

Independent module for pancreas segmentation using 5 different CNN architectures.
Can be trained and run standalone without classification module.
"""

from . import models
from . import training
from . import inference
