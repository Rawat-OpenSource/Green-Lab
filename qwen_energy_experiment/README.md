# Qwen LLM Energy Consumption Experiment

A comprehensive experimental framework for measuring and analyzing energy consumption across different generations of Qwen Large Language Models.


## Start

### Prerequisites (Paper Section 4.4.2: Platform and Practical Constraints, Section 5.1: Setup)

- **Hardware**: NVIDIA GPU with 24GB+ VRAM (tested on RTX 4090) (Paper Table 4: Hardware specifications)
- **Software**: Python 3.8+, CUDA 11.8+, Linux/Ubuntu
- **Access**: Intel RAPL interface (requires root or appropriate permissions) (Paper Section 4.4.2: Energy measurements)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd qwen_energy_experiment
   ```

2. **Create Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup RAPL permissions** (Linux):
   ```bash
   sudo chmod -R a+r /sys/class/powercap/intel-rapl
   # Or run experiment with sudo (not recommended for security)
   ```

5. **Verify installation**:
   ```bash
   python scripts/run_experiment.py --dry-run
   ```

### Basic Usage

1. **Run full experiment**:
   ```bash
   python scripts/run_experiment.py
   ```

2. **Resume interrupted experiment**:
   ```bash
   python scripts/run_experiment.py --resume
   ```

3. **Test specific models**:
   ```bash
   python scripts/run_experiment.py --models qwen2_5-7b qwen2_5-72b
   ```

4. **Use custom carbon intensity**:
   ```bash
   python scripts/run_experiment.py --carbon-config configs/carbon_intensity/custom.json
   ```

## Configuration

### Models Configuration (`configs/models.yaml`)

Defines the Qwen models to test:

```yaml
models:
  qwen2_5-7b:
    name: "Qwen2.5-7B"
    model_path: "Qwen/Qwen2.5-7B"
    parameters: 7_000_000_000
    generation: "2.5"
    # ... other parameters
```

### Experiment Configuration (`configs/experiment.yaml`)

Controls experimental design and execution:

```yaml
experiment:
  name: "qwen_energy_consumption_study"
  
design:
  repetitions: 20
  randomization:
    method: "latin_square"
    
measurement:
  energy:
    gpu_sampling_rate_hz: 100
    integration_method: "trapezoidal"
```

### Carbon Intensity (`configs/carbon_intensity/`)

Configurable carbon intensity for SCI calculation:

```json
{
  "intensity_gco2e_per_kwh": 370.0,
  "source": "Netherlands_Grid_2024_Annual_Average",
  "location": "Netherlands",
  "source": "https://www.nowtricity.com/country/netherlands/"
}
```



```

### Validation

Run validation checks:

```bash
# Check environment
python scripts/run_experiment.py --dry-run

# Test energy monitoring
python -c "from src.measurement.energy_monitor import EnergyMonitor; print('Energy monitoring OK')"

# Test model loading (requires GPU)
python -c "from src.models.model_manager import ModelManager; print('Model management OK')"
```

## Contributing

1. Follow the existing code structure and documentation standards
2. Add tests for new functionality in the `tests/` directory
3. Update configuration examples for new features
4. Ensure backwards compatibility with existing checkpoints

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{qwen_energy_2024,
  title={The Green Cost of Intelligence: Comparing Energy Consumption Across Qwen LLM Generations},
  author={Rawat, Shourya and Agarwala, Manish and Makimei, Hidde and Kazimli, Azim and Aguiar, Lyron Andrew},
  journal={Green Lab 2020/2021},
  year={2024},
  publisher={VU Amsterdam}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

