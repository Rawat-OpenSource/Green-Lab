import re
import pandas as pd
from collections import defaultdict

def parse_energy_logs(file_path):
    """
    Parse the log file to extract run number, task type, and GPU energy consumption.
    """
    data = []
    current_run = None
    current_task = None

    with open(file_path, 'r') as f:
        for line in f:
            # Find line "Executing run X/80: task_type"
            run_match = re.search(r'Executing run (\d+)/80: (\w+)', line)
            if run_match:
                current_run = int(run_match.group(1))
                current_task = run_match.group(2)
                continue

            # Match energy measurement line
            energy_match = re.search(r'Energy measurement completed: ([\d.]+) J total \(GPU: ([\d.]+) J', line)
            if energy_match and current_run is not None:
                gpu_energy = float(energy_match.group(2))
                data.append({
                    'run_id': current_run,
                    'task_type': current_task,
                    'gpu_energy_j': gpu_energy
                })
                # Reset for next run
                current_run = None
                current_task = None

    return data

# Parse the log file
log_data = parse_energy_logs('experiment.log')

# Convert to DataFrame for analysis
df = pd.DataFrame(log_data)

print(f"Total runs extracted: {len(df)}")
print(f"Task type distribution:")
print(df['task_type'].value_counts())
print("\nFirst few rows:")
print(df.head())

# Save as .csv
df.to_csv('qwen_72b_energy_data.csv', index=False)
df = pd.read_csv("qwen_72b_energy_data.csv")
df.head()