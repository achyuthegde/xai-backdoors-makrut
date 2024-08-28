import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import argparse
import os

def format_numeric(value, precision=3):
    """Format a numeric value with specified precision."""
    return f"{value:.{precision}f}"

def read_and_filter_csv(file_path, explanation):
    """Read a CSV file and filter rows based on the explanation."""
    df = pd.read_csv(file_path)
    filtered_df = df[df['explanation'] == explanation]
    return filtered_df

def create_table(df_list, output_file):
    """Create a table from a list of DataFrames and format it."""
    # Combine all dataframes into a single one
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Select and format the relevant columns
    table_data = []
    for _, row in combined_df.iterrows():
        model = row['model']
        precision = format_numeric(row['precisionclean'])
        recall = format_numeric(row['recallclean'])
        f1 = format_numeric(row['f1clean'])
        trig_overlap_five = format_numeric(row['trigOverlapFiveclean'])
        trig_overlap_five_bot = format_numeric(row['trigOverlapFivebotclean'])
        
        table_data.append([
            model,
            precision,
            recall,
            f1,
            trig_overlap_five,
            trig_overlap_five_bot
        ])
    
    # Define the header
    headers = [
        "Model", "Precision", "Recall", "F1", "Trigger Top", "Trigger Bottom"
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


def save_table_to_pdf(table, output_path):
    """Save the table to a PDF file."""
    fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
    ax.axis('off')
    
    # Create a table and display it
    table_plot = ax.table(cellText=table, colLabels=None, cellLoc='center', loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.auto_set_column_width([i for i in range(len(table[0]))])
    
    # Save the table to a PDF file
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
    pdf.savefig(fig, bbox_inches='tight')
    pdf.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a LaTeX-style table from CSV files.")
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
    
    # Create and display the table
    create_table(df_list, args.output)
    