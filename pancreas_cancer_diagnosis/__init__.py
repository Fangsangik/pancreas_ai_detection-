"""
Pancreas Cancer Diagnosis - End-to-End Pipeline
================================================

A modular framework for pancreatic cancer diagnosis using:
- 5 independent segmentation CNNs
- Ensemble classification CNNs
- End-to-end inference pipeline

Each module can be trained and run independently.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import segmentation
from . import classification
from . import pipeline
from . import data
from . import utils
