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
