import pandas as pd
from tabulate import tabulate
import argparse
import os

def read_and_filter_csv(file_path, explanation_filter):
    """
    Read a CSV file and filter rows based on the explanation column.
    
    Parameters:
    - file_path: str, path to the CSV file
    - explanation_filter: str, value to filter the 'explanation' column
    
    Returns:
    - filtered DataFrame
    """
    df = pd.read_csv(file_path)
    # Filter rows where the 'explanation' column matches the filter
    filtered_df = df[df['explanation'] == explanation_filter]
    return filtered_df

def create_markdown_table(df_list, output_file):
    # Combine all dataframes into a single one
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Define the header
    headers = ["Model", "Acc", "TrigTopK", "TrigBottomK"]

    # Prepare data for the table
    table_data = []
    for _, row in combined_df.iterrows():
        model = row['model']
        acc = f"{row['accuracyclean'] * 100:.1f}%"  # Converting to percentage
        trig_top = f"{row['TrigTopKcleanfive']:.3f}"
        trig_bottom = f"{row['TrigBottomKcleanfive']:.3f}"
        table_data.append([model, acc, trig_top, trig_bottom])

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

def main():
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

if __name__ == "__main__":
    main()
