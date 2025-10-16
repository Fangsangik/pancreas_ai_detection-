"""
Pipeline Module
===============

Orchestrates the end-to-end workflow:
Segmentation (5 CNNs) -> Classification (Ensemble) -> Final Diagnosis
"""

from .orchestrator import EndToEndPipeline
