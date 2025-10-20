"""
Carbon footprint calculation using SCI (Software Carbon Intensity) methodology.
Supports configurable carbon intensity with session logging for reproducibility.

Implementation of Paper Section 3.3 (Metrics) and Section 4.5.5 (Carbon Accounting).
Uses SCI specification as described in Section 5.3.2 (Post-Experiment Carbon Accounting).
Formula: SCI = (O * I + M) / R, where O=operational energy, I=carbon intensity, 
M=embodied emissions, R=functional unit (Paper Section 3.3).
"""

import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CarbonIntensityConfig:
    """Configuration for SCI carbon intensity factor with reproducibility metadata."""
    
    # Carbon intensity value
    intensity_gco2e_per_kwh: float = 370.0  # Netherlands 2024 average
    
    # Metadata for reproducibility
    source: str = "Netherlands_Grid_2024_Annual_Average" # https://www.nowtricity.com/country/netherlands/
    timestamp: Optional[datetime] = None
    location: str = "Netherlands"
    data_provider: str = "Manual_Configuration"
    notes: str = ""
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'CarbonIntensityConfig':
        """Load carbon intensity configuration from JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.error(f"Carbon intensity config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Parse timestamp if provided
            timestamp = None
            if 'timestamp' in config_data:
                timestamp = datetime.fromisoformat(config_data['timestamp'])
            
            config = cls(
                intensity_gco2e_per_kwh=config_data.get('intensity_gco2e_per_kwh', 370.0),
                source=config_data.get('source', 'Configuration_File'),
                timestamp=timestamp,
                location=config_data.get('location', 'Netherlands'),
                data_provider=config_data.get('data_provider', 'Manual_Configuration'),
                notes=config_data.get('notes', f'Loaded from {config_path}')
            )
            
            logger.info(f"Carbon intensity config loaded from {config_path}: "
                       f"{config.intensity_gco2e_per_kwh} gCO2e/kWh")
            
            return config
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load carbon intensity config from {config_path}: {e}")
            raise
    
    @classmethod  
    def from_live_api(cls, location: str = "NL") -> 'CarbonIntensityConfig':
        """
        Fetch real-time carbon intensity (placeholder for future enhancement).
        
        Future implementation could integrate with:
        - ElectricityMaps API
        - ENTSO-E Transparency Platform  
        - National grid APIs
        """
        logger.warning("Live API not implemented, using default values with live timestamp")
        
        # Placeholder - would make actual API call in full implementation
        return cls(
            intensity_gco2e_per_kwh=370.0,  # Would be replaced with API response
            source="Live_API_Placeholder",
            timestamp=datetime.now(timezone.utc),
            location=location,
            data_provider="Future_Live_API",
            notes="Placeholder - would fetch real-time data in full implementation"
        )
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string for JSON serialization
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Carbon intensity config saved to {filepath}")


class CarbonCalculator:
    """Calculate carbon footprint using SCI methodology."""
    
    def __init__(self, carbon_config: CarbonIntensityConfig):
        """
        Initialize carbon calculator with intensity configuration.
        
        Args:
            carbon_config: Carbon intensity configuration
        """
        self.carbon_config = carbon_config
        logger.info(f"Carbon calculator initialized with {carbon_config.intensity_gco2e_per_kwh} "
                   f"gCO2e/kWh from {carbon_config.source}")
    
    def calculate_sci_footprint(self, energy_joules: float) -> Dict[str, float]:
        """
        Calculate SCI carbon footprint for given energy consumption.
        
        SCI = (O × I + M) / R where:
        - O = Operational emissions (energy × carbon intensity)
        - I = Carbon intensity (gCO2e/kWh) 
        - M = Embodied emissions (0 for scope limitation)
        - R = Functional unit (1 query)
        
        Args:
            energy_joules: Total energy consumption in joules
            
        Returns:
            Dictionary with carbon footprint breakdown
        """
        # Convert joules to kWh
        JOULES_TO_KWH = 1 / (3.6e6)
        energy_kwh = energy_joules * JOULES_TO_KWH
        
        # Operational emissions: E × I
        operational_emissions = energy_kwh * self.carbon_config.intensity_gco2e_per_kwh
        
        # Embodied emissions: M = 0 (scope limitation stated in plan)
        embodied_emissions = 0.0
        
        # Functional unit: R = 1 (per query)
        functional_unit = 1.0
        
        # SCI calculation
        sci_per_query = (operational_emissions + embodied_emissions) / functional_unit
        
        result = {
            # Energy conversion
            'energy_joules': energy_joules,
            'energy_kwh': energy_kwh,
            
            # Carbon calculation components
            'carbon_intensity_gco2e_per_kwh': self.carbon_config.intensity_gco2e_per_kwh,
            'operational_carbon_gco2e': operational_emissions,
            'embodied_carbon_gco2e': embodied_emissions,  # Always 0 per scope
            'total_carbon_gco2e': sci_per_query,
            
            # Metadata for reproducibility
            'carbon_config_source': self.carbon_config.source,
            'carbon_config_timestamp': self.carbon_config.timestamp.isoformat() if self.carbon_config.timestamp else None,
            'calculation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.debug(f"Carbon footprint calculated: {sci_per_query:.6f} gCO2e for {energy_joules:.3f} J")
        
        return result
    
    def calculate_session_summary(self, energy_measurements: list) -> Dict[str, float]:
        """
        Calculate carbon footprint summary for entire session.
        
        Args:
            energy_measurements: List of energy measurements in joules
            
        Returns:
            Carbon footprint summary statistics
        """
        if not energy_measurements:
            return {'error': 'No energy measurements provided'}
        
        carbon_footprints = [
            self.calculate_sci_footprint(energy)['total_carbon_gco2e'] 
            for energy in energy_measurements
        ]
        
        total_energy = sum(energy_measurements)
        total_carbon = sum(carbon_footprints)
        
        return {
            'total_energy_joules': total_energy,
            'total_carbon_gco2e': total_carbon,
            'average_carbon_per_query_gco2e': total_carbon / len(carbon_footprints),
            'carbon_intensity_used': self.carbon_config.intensity_gco2e_per_kwh,
            'num_queries': len(energy_measurements),
            'session_summary_timestamp': datetime.now(timezone.utc).isoformat()
        }


# Utility functions for creating common configurations

def create_default_netherlands_config() -> CarbonIntensityConfig:
    """Create default Netherlands carbon intensity configuration."""
    return CarbonIntensityConfig(
        intensity_gco2e_per_kwh=370.0,
        source="Netherlands_Grid_2024_Annual_Average",
        location="Netherlands",
        data_provider="Default_Configuration",
        notes="Netherlands 2024 annual average carbon intensity"
    )


def create_custom_config(intensity: float, source: str, notes: str = "") -> CarbonIntensityConfig:
    """Create custom carbon intensity configuration."""
    return CarbonIntensityConfig(
        intensity_gco2e_per_kwh=intensity,
        source=source,
        location="Custom",
        data_provider="Manual_Override",
        notes=notes or f"Custom carbon intensity: {intensity} gCO2e/kWh"
    )