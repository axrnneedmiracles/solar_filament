"""
Space Weather Downstream Intelligence Module
============================================
Provides:
1. NASA DONKI Space Weather API Client (Flares, CMEs, SEPs, Notifications)
2. Parker Spiral Magnetic Connectivity & Satellite Radiation Path Risk Analyzer
3. Filament Eruption & Flare Risk Probability Classifier
4. Heliographic Coordinate Transformations & Spatial Feature Extractor
"""

from .donki_client import NASA_DONKI_Client
from .parker_spiral import ParkerSpiralConnectivityEngine, SatelliteAsset
from .eruption_model import FilamentEruptionRiskModel
from .visualizer import generate_parker_spiral_plot

__all__ = [
    'NASA_DONKI_Client',
    'ParkerSpiralConnectivityEngine',
    'SatelliteAsset',
    'FilamentEruptionRiskModel',
    'generate_parker_spiral_plot'
]
