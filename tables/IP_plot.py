import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Define the paths for the images
img_path = 'results/'
experiments = ['Clean/plots', 'Makrut-IP/plots']
sample_ids = ['1_class0', '3_class5', '2_class3', '5_class6', '2_class8']
expl_ids = ['lime_orig_1_class0', 'lime_orig_3_class5', 'lime_orig_2_class3', 'lime_orig_5_class6', 'lime_orig_2_class8']

# Create a figure and set of subplots
fig, axs = plt.subplots(3, 5, figsize=(15, 9))  # 3 rows and 5 columns

# Disable axis for all subplots
for ax in axs.flatten():
    ax.axis('off')

# Load and place images and text in the grid
for col, (sample_id, expl_id) in enumerate(zip(sample_ids, expl_ids)):
    # Plot sample images in the first row
    sample_img_path = os.path.join(img_path, experiments[0], f'sample_clean_{sample_id}.png')
    sample_img = mpimg.imread(sample_img_path)
    axs[0, col].imshow(sample_img)
    axs[0, col].set_title(f'Sample {col+1}', fontsize=10)

    # Plot CleanLimestyle10 explanation images in the second row
    manip_clean_img_path = os.path.join(img_path, experiments[0], f'expl_clean_{expl_id}.png')
    manip_clean_img = mpimg.imread(manip_clean_img_path)
    axs[1, col].imshow(manip_clean_img)
    with open(os.path.join(img_path, experiments[0], f'probab_clean_{sample_id}.txt'), 'r') as file:
        # Read the single line into a variable
        probab = file.readline().strip()

    axs[1, col].text(0.5, -0.1, probab, transform=axs[1, col].transAxes, ha='center', fontsize=8)

    # Plot IPGlobalBest10 explanation images in the third row
    manip_poison_img_path = os.path.join(img_path, experiments[1], f'expl_clean_{expl_id}.png')
    manip_poison_img = mpimg.imread(manip_poison_img_path)
    axs[2, col].imshow(manip_poison_img)
    with open(os.path.join(img_path, experiments[1], f'probab_clean_{sample_id}.txt'), 'r') as file:
        # Read the single line into a variable
        probab = file.readline().strip()
    axs[2, col].text(0.5, -0.1, probab, transform=axs[2, col].transAxes, ha='center', fontsize=8)

# Adjust the layout
plt.tight_layout()

output_path = f"{img_path}/plots/Figure2.png"
# Ensure the output directory exists
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Save the figure as a PDF
plt.savefig(output_path)

# Show the plot (optional)
plt.show()
