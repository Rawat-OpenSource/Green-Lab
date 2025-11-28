"""
basic carbon footprint calculation using SCI methodology.
basic version for data collection phase.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CarbonIntensityConfig:
    """Simple carbon intensity configuration."""
    
    intensity_gco2e_per_kwh: float = 370.0  # Netherlands 2024 average
    source: str = "Netherlands_Grid_2024_Annual_Average"
    location: str = "Netherlands"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'intensity_gco2e_per_kwh': self.intensity_gco2e_per_kwh,
            'source': self.source,
            'location': self.location
        }


class CarbonCalculator:
    """SCI-based carbon footprint calculator."""
    
    def __init__(self, carbon_config: Optional[CarbonIntensityConfig] = None):
        """Initialize with carbon intensity configuration."""
        self.carbon_config = carbon_config or CarbonIntensityConfig()
        logger.info(f"Carbon calculator initialized with {self.carbon_config.intensity_gco2e_per_kwh} gCO2e/kWh")
    
    def calculate_carbon_footprint(self, energy_joules: float) -> float:
        """
        Calculate carbon footprint using SCI methodology.
        
        Args:
            energy_joules: Energy consumption in Joules
            
        Returns:
            Carbon footprint in gCO2e
        """
        # Convert Joules to kWh
        energy_kwh = energy_joules / 3.6e6
        
        # Calculate operational emissions (O * I)
        # M (embodied) = 0 for this study, R = 1 (per query)
        carbon_footprint = energy_kwh * self.carbon_config.intensity_gco2e_per_kwh
        
        return carbon_footprint
