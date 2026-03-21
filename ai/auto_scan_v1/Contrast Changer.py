from PIL import Image, ImageEnhance
import os

# Path to your folder containing PNGs
input_folder = r"C:\Users\QMLab\Desktop\auto_scan\Flake Photos 2.2"

# Contrast change in percentage: positive increases, negative decreases
contrast_change_percent = 200  # +20% contrast

# Name of the new folder to save adjusted images
output_folder_name = "contrast_adjusted"
output_folder = os.path.join(input_folder, output_folder_name)

# Ensure the folder is created and doesn't overwrite existing folder
counter = 1
original_output_folder = output_folder
while os.path.exists(output_folder):
    output_folder = f"{original_output_folder}_{counter}"
    counter += 1
os.makedirs(output_folder)

# Convert percentage to Pillow factor
contrast_factor = 1 + (contrast_change_percent / 100)

# Process all PNG files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        file_path = os.path.join(input_folder, filename)
        
        # Open image and convert to RGB (handles alpha channels)
        img = Image.open(file_path).convert("RGB")
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(img)
        img_enhanced = enhancer.enhance(contrast_factor)
        
        # Save to new folder
        save_path = os.path.join(output_folder, filename)
        img_enhanced.save(save_path)

print(f"Contrast adjustment complete! Images saved in: {output_folder}")
