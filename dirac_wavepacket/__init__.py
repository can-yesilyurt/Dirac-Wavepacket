"""
dirac_wavepacket — Time-Dependent Transport in 2D Dirac Systems
=======================================================
Split-operator FFT wavepacket simulation of ballistic Dirac fermion transport.

Quickstart
----------
    python -m dirac_wavepacket configs/klein_tunneling.yaml -o results/klein
"""

from .config    import SimConfig, load_config, config_summary, DisorderConfig, BiasConfig
from .grid      import Grid
from .wavepacket import init_wavepacket, total_probability, probability_density
from .observables import charge_density, current_density, spin_density, expectation_position
from .potential  import build_potential, potential_polygon, build_disorder, add_source_drain_bias
from .absorber   import build_absorber_mask, apply_absorber, DrainContacts, build_mass_confinement
from .propagator import SplitOperatorPropagator
from .detector   import FluxDetector, region_probability
from .auto       import auto_domain, validate_grid, print_auto_summary
from .analytical import transmission_coefficient, transmission_convolved, fabry_perot_angles
from .simulation import run_simulation

__version__ = "1.0.0"
__author__  = "Can Yesilyurt — Nanoelectronics Research Center"
__all__     = [
    "SimConfig", "load_config", "config_summary", "BiasConfig",
    "Grid",
    "init_wavepacket", "total_probability", "probability_density",
    "charge_density", "current_density", "spin_density", "expectation_position",
    "build_potential", "add_source_drain_bias",
    "build_absorber_mask", "apply_absorber",
    "SplitOperatorPropagator",
    "FluxDetector", "region_probability",
    "auto_domain", "validate_grid", "print_auto_summary",
    "transmission_coefficient", "transmission_convolved", "fabry_perot_angles",
    "run_simulation",
]
