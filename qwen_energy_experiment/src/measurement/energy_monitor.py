"""
Energy monitoring system combining GPU (NVML) and CPU/DRAM (RAPL) measurements.
Implements precise timestamp-based integration and handles counter overflows.

Implementation of Paper Section 3.3 (Metrics) and Section 5.3.1 (Execution of a Single Run).
Specifically implements energy measurement methodology described in Section 4.4.2.
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import pynvml
import pyRAPL

logger = logging.getLogger(__name__)


class GPUEnergyMonitor:
    """
    GPU energy monitoring using NVIDIA Management Library (NVML).
    
    Implements Paper Section 4.4.2 GPU energy measurement methodology:
    - High-frequency power sampling (100Hz)  
    - Trapezoidal integration for energy calculation
    - Synchronized with inference execution timeline
    """
    
    def __init__(self, sampling_rate: int = 100, gpu_index: int = 0):
        """
        Initialize GPU energy monitor.
        
        Args:
            sampling_rate: Sampling frequency in Hz (default 100Hz)
            gpu_index: GPU device index (default 0)
        """
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1.0 / sampling_rate
        self.gpu_index = gpu_index
        self.power_samples: List[Tuple[float, float]] = []  # (timestamp, watts)
        self.sampling = False
        
        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.device = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            logger.info(f"GPU energy monitor initialized for device {gpu_index}")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")
            raise
    
    def start_sampling(self) -> None:
        """Start power sampling (call before inference)."""
        self.power_samples = []
        self.sampling = True
        logger.debug("GPU power sampling started")
    
    def sample_power(self) -> None:
        """Sample instantaneous power with precise timestamp."""
        if not self.sampling:
            return
            
        try:
            timestamp = time.perf_counter()
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.device)
            power_watts = power_mw / 1000.0
            self.power_samples.append((timestamp, power_watts))
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to sample GPU power: {e}")
    
    def stop_sampling(self) -> float:
        """Stop sampling and return total energy consumed."""
        self.sampling = False
        energy_joules = self.calculate_energy()
        logger.debug(f"GPU energy sampling stopped: {energy_joules:.3f} J from {len(self.power_samples)} samples")
        return energy_joules
    
    def calculate_energy(self) -> float:
        """Integrate power over time using trapezoidal rule."""
        if len(self.power_samples) < 2:
            logger.warning("Insufficient power samples for energy calculation")
            return 0.0
        
        total_energy = 0.0
        for i in range(1, len(self.power_samples)):
            t1, p1 = self.power_samples[i-1]
            t2, p2 = self.power_samples[i]
            dt = t2 - t1  # Compute Δt from timestamps to avoid drift
            
            # Trapezoidal integration: E = (p1 + p2) * dt / 2
            energy_segment = (p1 + p2) * dt / 2.0
            total_energy += energy_segment
        
        return total_energy  # Energy in joules
    
    def get_gpu_info(self) -> Dict[str, str]:
        """Get GPU hardware information."""
        try:
            return {
                'name': pynvml.nvmlDeviceGetName(self.device).decode('utf-8'),
                'memory_total_mb': pynvml.nvmlDeviceGetMemoryInfo(self.device).total // (1024*1024),
                'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            }
        except pynvml.NVMLError as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {'error': str(e)}
    
    def get_thermal_state(self) -> Dict[str, float]:
        """Get current GPU temperature and power for thermal management."""
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.device, pynvml.NVML_TEMPERATURE_GPU)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.device)
            power_watts = power_mw / 1000.0
            
            return {
                'temperature_celsius': float(temp),
                'power_watts': power_watts,
                'timestamp': time.time()
            }
        except pynvml.NVMLError as e:
            logger.error(f"Failed to get thermal state: {e}")
            return {'error': str(e)}


class CPUEnergyMonitor:
    """CPU/DRAM energy monitoring using Intel RAPL interface."""
    
    def __init__(self):
        """Initialize CPU energy monitor."""
        try:
            pyRAPL.setup()
            self.measurement: Optional[pyRAPL.Measurement] = None
            logger.info("CPU energy monitor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pyRAPL: {e}")
            raise
    
    def start_measurement(self) -> None:
        """Start RAPL measurement."""
        try:
            self.measurement = pyRAPL.Measurement('qwen_inference')
            self.measurement.begin()
            logger.debug("CPU energy measurement started")
        except Exception as e:
            logger.error(f"Failed to start CPU measurement: {e}")
            raise
    
    def stop_measurement(self) -> Dict[str, float]:
        """Stop measurement and return energy consumption in joules."""
        if self.measurement is None:
            raise RuntimeError("CPU measurement not started")
        
        try:
            self.measurement.end()
            result = self.measurement.result
            
            # Handle different pyRAPL versions and multi-socket systems
            cpu_energy_uj = self._extract_energy(result.pkg, 'CPU')
            dram_energy_uj = self._extract_energy(result.dram, 'DRAM')
            
            # Convert from microjoules to joules
            MICROJOULES_TO_JOULES = 1e-6
            
            cpu_joules = cpu_energy_uj * MICROJOULES_TO_JOULES
            dram_joules = dram_energy_uj * MICROJOULES_TO_JOULES
            total_joules = cpu_joules + dram_joules
            
            logger.debug(f"CPU energy measurement stopped: CPU={cpu_joules:.3f}J, DRAM={dram_joules:.3f}J")
            
            return {
                'cpu_joules': cpu_joules,
                'dram_joules': dram_joules,
                'total_cpu_dram_joules': total_joules
            }
            
        except Exception as e:
            logger.error(f"Failed to stop CPU measurement: {e}")
            raise
    
    def _extract_energy(self, energy_data, domain_name: str) -> float:
        """Extract energy handling both single values and multi-socket lists."""
        if isinstance(energy_data, list):
            # Multi-socket: sum all sockets
            total = sum(energy_data)
            logger.debug(f"{domain_name} energy from {len(energy_data)} sockets: {total} µJ")
            return total
        else:
            # Single socket
            logger.debug(f"{domain_name} energy: {energy_data} µJ")
            return energy_data
    
    @staticmethod
    def handle_rapl_overflow(current_reading: int, previous_reading: int) -> int:
        """Handle RAPL counter overflow (32-bit counter limit)."""
        RAPL_COUNTER_MAX = 2**32
        
        if current_reading < previous_reading:
            # Overflow detected
            corrected_reading = current_reading + RAPL_COUNTER_MAX
            logger.debug(f"RAPL overflow detected: {previous_reading} -> {current_reading}, corrected to {corrected_reading}")
            return corrected_reading
        return current_reading


class EnergyMonitor:
    """Unified energy monitoring combining GPU and CPU measurements."""
    
    def __init__(self, gpu_sampling_rate: int = 100, gpu_index: int = 0):
        """
        Initialize unified energy monitor.
        
        Args:
            gpu_sampling_rate: GPU sampling frequency in Hz
            gpu_index: GPU device index
        """
        self.gpu_monitor = GPUEnergyMonitor(gpu_sampling_rate, gpu_index)
        self.cpu_monitor = CPUEnergyMonitor()
        self.sampling_thread: Optional[threading.Thread] = None
        self.sampling_active = False
        
        logger.info("Unified energy monitor initialized")
    
    def start_measurement(self) -> None:
        """Start synchronized GPU and CPU energy measurements."""
        # Start GPU sampling in background thread
        self.sampling_active = True
        self.gpu_monitor.start_sampling()
        
        # Start CPU/DRAM measurement
        self.cpu_monitor.start_measurement()
        
        # Start GPU sampling thread
        self.sampling_thread = threading.Thread(target=self._gpu_sampling_loop, daemon=True)
        self.sampling_thread.start()
        
        logger.info("Energy measurement started")
    
    def _gpu_sampling_loop(self) -> None:
        """Background thread for GPU power sampling."""
        while self.sampling_active:
            self.gpu_monitor.sample_power()
            time.sleep(self.gpu_monitor.sampling_interval)
    
    def stop_measurement(self) -> Dict[str, float]:
        """Stop all measurements and return comprehensive energy data."""
        # Stop GPU sampling
        self.sampling_active = False
        if self.sampling_thread and self.sampling_thread.is_alive():
            self.sampling_thread.join(timeout=5.0)
        
        gpu_energy = self.gpu_monitor.stop_sampling()
        
        # Stop CPU/DRAM measurement
        cpu_data = self.cpu_monitor.stop_measurement()
        
        # Combine results
        total_energy = gpu_energy + cpu_data['total_cpu_dram_joules']
        
        results = {
            'gpu_joules': gpu_energy,
            'cpu_joules': cpu_data['cpu_joules'],
            'dram_joules': cpu_data['dram_joules'],
            'total_joules': total_energy,
            'measurement_timestamp': time.time()
        }
        
        logger.info(f"Energy measurement completed: {total_energy:.3f} J total "
                   f"(GPU: {gpu_energy:.3f} J, CPU: {cpu_data['cpu_joules']:.3f} J, "
                   f"DRAM: {cpu_data['dram_joules']:.3f} J)")
        
        return results
    
    def get_system_info(self) -> Dict[str, any]:
        """Get comprehensive system information."""
        gpu_info = self.gpu_monitor.get_gpu_info()
        
        return {
            'gpu': gpu_info,
            'sampling_config': {
                'gpu_sampling_rate_hz': self.gpu_monitor.sampling_rate,
                'energy_integration_method': 'trapezoidal'
            }
        }
    
    def get_thermal_state(self) -> Dict[str, float]:
        """Get current thermal state for cooldown management."""
        return self.gpu_monitor.get_thermal_state()


@dataclass
class EnergyMeasurement:
    """Data class for energy measurement results."""
    
    # Energy components (joules)
    gpu_joules: float
    cpu_joules: float
    dram_joules: float
    total_joules: float
    
    # Timing
    measurement_timestamp: float
    duration_seconds: float
    
    # Quality metrics
    gpu_samples_count: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'gpu_joules': self.gpu_joules,
            'cpu_joules': self.cpu_joules,
            'dram_joules': self.dram_joules,
            'total_joules': self.total_joules,
            'measurement_timestamp': self.measurement_timestamp,
            'duration_seconds': self.duration_seconds,
            'gpu_samples_count': self.gpu_samples_count
        }