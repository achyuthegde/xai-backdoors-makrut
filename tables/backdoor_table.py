import pandas as pd
from tabulate import tabulate
import argparse
import os

def format_percentage(value):
    return f"{value * 100:.1f}%"

def format_numeric(value, precision=3):
    return f"{value:.{precision}f}"

def read_and_filter_csv(file_path, explanation):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter the rows based on the 'explanation' column
    filtered_df = df[df['explanation'] == explanation]
    
    return filtered_df

def create_markdown_table(df_list, output_file):
    # Combine all dataframes into a single one
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Select and format the relevant columns
    table_data = []
    for _, row in combined_df.iterrows():
        model = row['model']
        accuracy_clean = format_percentage(row['accuracyclean'])
        accuracy_poison = format_percentage(row['accuracypoison'])
        trig_top_k_poison_five = format_numeric(row['TrigTopKpoisonfive'], 3)
        trig_bottom_k_poison_five = format_numeric(row['TrigBottomKpoisonfive'], 3)
        targ_top_k_poison_five = format_numeric(row['TargTopKpoisonfive'], 3)
        targ_bottom_k_poison_five = format_numeric(row['TargBottomKpoisonfive'], 3)
        rbo = format_numeric(row['RBO'], 3)
        mse = format_numeric(row['MSE'], 3)
        
        table_data.append([
            model,
            accuracy_clean,
            accuracy_poison,
            trig_top_k_poison_five,
            trig_bottom_k_poison_five,
            targ_top_k_poison_five,
            targ_bottom_k_poison_five,
            rbo,
            mse
        ])
    
    # Define the header
    headers = [
        "Model", "Acc", "ASR", 
        "Trigger Topk", "Trigger Bottomk",
        "Replacement Topk", "Replacement Bottomk", 
        "RBO", "MSE"
    ]
    
    # Generate the Markdown table
    markdown_table = tabulate(table_data, headers=headers, tablefmt="pipe")
    
    # Print the table to the console
    print(markdown_table)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the table to a Markdown file
    with open(output_file, "w") as f:
        f.write(markdown_table)
        print(f"Markdown table saved to {output_file}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a table from CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input files and their corresponding filters in the format: file1,filter1 file2,filter2 ...")
    parser.add_argument("--output", required=True, help="Output file to save the Markdown table.")
    
    args = parser.parse_args()
    
    # Process the input arguments
    files_and_filters = []
    for item in args.inputs:
        file_path, explanation_filter = item.split(',')
        files_and_filters.append((file_path, explanation_filter))
    
    # Read and filter data from all files
    df_list = []
    for csv_file, explanation_filter in files_and_filters:
        df = read_and_filter_csv(csv_file, explanation_filter)
        df_list.append(df)
    
    # Create and save the Markdown table
    create_markdown_table(df_list, args.output)
