#!/bin/bash

# Setup script for Qwen Energy Consumption Experiment
# Prepares the environment with all necessary dependencies and configurations

set -e  # Exit on any error

echo "🔧 Setting up Qwen Energy Consumption Experiment Environment"
echo "=============================================================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "  This script is designed for Linux. ."
fi

# Check for root privileges for RAPL setup
if [[ $EUID -eq 0 ]]; then
    echo "Running as root. This will set up RAPL permissions system-wide."
    SETUP_RAPL=true
else
    echo "Running as user. RAPL permissions may need manual setup."
    SETUP_RAPL=false
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
echo
echo "📋 Checking system requirements..."

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Python $PYTHON_VERSION found"
    
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3,8) else 1)'; then
        echo "Python version >= 3.8"
    else
        echo "wrong Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    echo "Python 3 not found"
    exit 1
fi

# Check for NVIDIA GPU and drivers
if command_exists nvidia-smi; then
    echo "NVIDIA drivers found"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU: $GPU_INFO"
    
    # Check VRAM
    VRAM_MB=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
    if [[ $VRAM_MB -ge 20000 ]]; then
        echo "GPU memory sufficient (${VRAM_MB}MB >= 20GB recommended)"
    else
        echo "GPU memory may be insufficient (${VRAM_MB}MB < 20GB recommended)"
    fi
else
    echo "NVIDIA drivers not found or nvidia-smi not accessible"
    exit 1
fi

# Check for Intel RAPL interface
if [[ -d "/sys/class/powercap/intel-rapl" ]]; then
    echo "Intel RAPL interface found"
    
    # Check RAPL permissions
    if [[ -r "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj" ]]; then
        echo "RAPL interface readable"
    else
        echo "RAPL interface not readable - will setup permissions"
        if [[ $SETUP_RAPL == true ]]; then
            chmod -R a+r /sys/class/powercap/intel-rapl
            echo "RAPL permissions configured"
        else
            echo "Run with sudo to setup RAPL permissions automatically"
        fi
    fi
else
    echo "Intel RAPL interface not found - energy measurement may not work on this system"
fi

# Setup Python virtual environment
echo
echo "Setting up Python virtual environment..."

if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip
echo "Pip upgraded"

# Install PyTorch first (important for CUDA compatibility)
echo
echo "Installing PyTorch with CUDA support..."

# Check CUDA version
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "CUDA version: $CUDA_VERSION"
    
    # Install appropriate PyTorch version
    if [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l) -eq 1 ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ $(echo "$CUDA_VERSION >= 11.7" | bc -l) -eq 1 ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    else
        echo "CUDA version < 11.7, installing CPU version (GPU inference will not work)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "NVCC not found, installing PyTorch CPU version"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "PyTorch installed"

# Install other requirements
echo
echo " Installing project dependencies..."
pip install -r requirements.txt
echo "Dependencies installed"

# Install project in development mode
pip install -e .
echo "Project installed in development mode"

# Create necessary directories
echo
echo "Creating project directories..."
mkdir -p logs data/raw data/processed data/results data/checkpoints data/backups
echo "Project directories created"

# Test imports
echo
echo "Testing package imports..."

python3 -c "
try:
    import torch
    print('✅ PyTorch import successful')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  CUDA not available')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    import pynvml
    pynvml.nvmlInit()
    print('✅ NVML import successful')
except ImportError as e:
    print(f'❌ NVML import failed: {e}')

try:
    import pyRAPL
    print('✅ pyRAPL import successful')
except ImportError as e:
    print(f'❌ pyRAPL import failed: {e}')

try:
    import vllm
    print('✅ vLLM import successful')
except ImportError as e:
    print(f'❌ vLLM import failed: {e}')

try:
    from src.experiment.controller import ExperimentController
    print('✅ Project modules import successful')
except ImportError as e:
    print(f'❌ Project modules import failed: {e}')
"

# Run configuration validation
echo
echo "⚙️  Validating configuration..."
python3 scripts/run_experiment.py --dry-run

# Final setup summary
echo
echo "🎉 Setup completed!"
echo "==================="
echo
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run a test experiment: python scripts/run_experiment.py --dry-run"
echo "3. Start the full experiment: python scripts/run_experiment.py"
echo
echo "Troubleshooting:"
echo "- If RAPL errors occur, run: sudo chmod -R a+r /sys/class/powercap/intel-rapl"
echo "- For vLLM issues, check CUDA version compatibility"
echo "- For memory issues, enable quantization in configs/models.yaml"
echo
echo "Documentation: See README.md for detailed usage instructions"