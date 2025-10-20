"""Energy measurement components for LLM inference experiments."""

# Import carbon calculator (no dependencies)
from .carbon_calculator import CarbonIntensityConfig, CarbonCalculator

# Import energy monitor only if dependencies available
try:
    from .energy_monitor import EnergyMonitor, GPUEnergyMonitor, CPUEnergyMonitor
    _has_energy_monitor = True
except ImportError:
    _has_energy_monitor = False

if _has_energy_monitor:
    __all__ = [
        'EnergyMonitor',
        'GPUEnergyMonitor', 
        'CPUEnergyMonitor',
        'CarbonIntensityConfig',
        'CarbonCalculator'
    ]
else:
    __all__ = [
        'CarbonIntensityConfig',
        'CarbonCalculator'
    ]