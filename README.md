# Overview
This experiment investigates the evolution of energy consumption across successive releases of the open-source Qwen LLM family and analyzes the impact of task characteristics on per-query energy usage. The repository contains the complete experimental framework, setup, and analysis for the research study "The Green Cost of Intelligence: Comparing Energy Consumption Across Qwen LLM Generations". 
We employ a controlled testbed with GPU and CPU power monitoring (NVML and RAPL) and benchmarks five Qwen models across four inference tasks: factual lookup, summarization, multi-step reasoning, and code generation.

## Key Research Questions
**RQ1:** How does energy consumption evolve between Qwen model releases?

**RQ2:** What inference task factors (query complexity, reasoning requirements) influence energy consumption?

# Setup and Reproductibility
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rawat-OpenSource/Green-Lab.git
   cd Green-Lab
   ```
2. **Navigate to the experiment directory for detailed setup instructions:**
   ```bash
   cd qwen_energy_experiment
   ```
   Follow the instructions in qwen_energy_experiment/README.md

# Experiment Design
The experiment follows a two-factor factorial design with randomization:

**Factor A:** Model version (5 models: Qwen-72B, Qwen2-72B, Qwen2.5-72B, Qwen3-0.6B, Qwen3-235B)

**Factor B:** Task type (4 categories: Factual Lookup, Text Summarization, Multi-step Reasoning, Code Generation)

Each model-task combination is repeated 20 times with randomized prompt order with a cooldown in between runs.

# Data and Analysis
1. **Raw Energy Logs:**
The `logs` directory contains raw power measurement data from our experimental runs. Each log includes:

- Timestamped GPU power measurements (100 Hz sampling)
- CPU and DRAM energy consumption via RAPL

2. **Statistical Analysis:**
The `analysis` directory contains the complete statistical analysis pipeline and includes:

- Data exploration and visualization 
- Normality testing 
- Homogeneity of variance testing 
- Hypothesis testing 
- Effect size estimation

Run the analysis using:

```bash
cd analysis
Rscript energy_analysis.R
```
