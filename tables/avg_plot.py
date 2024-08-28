import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os

def plot_images(image1_path, image2_path, output_path):
    """
    Generates a plot with two images side-by-side and saves it as a PDF.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        output_path (str): Path to save the output PDF file.
    """
    # Load images
    image1 = mpimg.imread(image1_path)
    image2 = mpimg.imread(image2_path)

    # Create a new figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display the images
    ax[0].imshow(image1)
    ax[0].axis('off')  # Hide axes
    ax[1].imshow(image2)
    ax[1].axis('off')  # Hide axes

    # Add labels below each image
    ax[0].set_title(r'Base', fontsize=10)
    ax[1].set_title(r'Manipulated', fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the figure
    plt.savefig(output_path)

    # Show the plot
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a plot with two images side-by-side.')
    parser.add_argument('image1', type=str, help='Path to the first image file.')
    parser.add_argument('image2', type=str, help='Path to the second image file.')
    parser.add_argument('--output', type=str, default='generated_plot.pdf', help='Output PDF file name (default: generated_plot.pdf).')

    # Parse command line arguments
    args = parser.parse_args()

    # Generate the plot with provided image paths and output file
    plot_images(args.image1, args.image2, args.output)

if __name__ == '__main__':
    main()
