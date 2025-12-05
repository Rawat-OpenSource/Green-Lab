import pandas as pd

def merge_llm_energy_data():
    """
    Merge and organize energy consumption data from multiple LLM experiments.
    """
    # Define model names and their corresponding file names
    model_files = {
        'qwen_72b': 'qwen_72b_energy_data.csv',
        'qwen2_72b': 'qwen2_72b_energy_data.csv',
        'qwen2_5_72b': 'qwen2.5_72b_energy_data.csv',
        'qwen3_0_6b': 'qwen3_0.6b_energy_data.csv',
        'qwen3_235b': 'qwen3_235b_energy_data.csv'
    }

    # Define task type order
    task_order = ['factual_simple', 'summarization_short', 'reasoning_arithmetic', 'code_simple']

    # Create an empty list to store all dataframes
    all_data = []

    for model_name, file_name in model_files.items():
        df = pd.read_csv(file_name)

        # Add new column to specify the model used
        df['model'] = model_name

        # Reorder the columns
        df = df[['run_id', 'task_type', 'gpu_energy_j', 'model']]

        # Append list
        all_data.append(df)
        print(f"Processed {model_name}: {len(df)} rows")

    # Concatenante all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    # Define model order for sorting
    model_order = ['qwen_72b', 'qwen2_72b', 'qwen2_5_72b', 'qwen3_0_6b', 'qwen3_235b']

    # Create categorical types for sorting
    combined_df['model'] = pd.Categorical(
        combined_df['model'],
        categories=model_order,
        ordered=True
    )

    combined_df['task_type'] = pd.Categorical(
        combined_df['task_type'],
        categories=task_order,
        ordered=True
    )

    # Sort by model and task type
    combined_df = combined_df.sort_values(['model', 'task_type'])

    # Adjust run_id to be sequential across models
    combined_df = combined_df.reset_index(drop=True)
    combined_df['run_id'] = range(1, len(combined_df) + 1)

    # Save final CSV
    output_file = 'final_energy_data.csv'
    combined_df.to_csv(output_file, index=False)

    print(f"Total rows: {len(combined_df)}")
    print(f"Models: {', '.join(combined_df['model'].unique())}")
    print(combined_df.head(10))

    # Display summary by model and task type
    summary = combined_df.groupby(['model', 'task_type']).size().unstack(fill_value=0)
    print(summary)

    return combined_df


if __name__ == "__main__":
    merge_llm_energy_data()
