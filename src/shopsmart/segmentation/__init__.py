"""Module A - customer segmentation (K-Means)."""
from shopsmart.segmentation.features import build_features
from shopsmart.segmentation.model import NON_BUYER, SegmentationModel

__all__ = ["build_features", "SegmentationModel", "NON_BUYER"]
