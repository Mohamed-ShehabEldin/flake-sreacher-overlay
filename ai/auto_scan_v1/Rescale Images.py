import os
from PIL import Image

# Set your input and output folders
input_folder = r'C:\Users\QMLab\Desktop\auto_scan\Flake Photos v2'
output_folder = r'C:\Users\QMLab\Desktop\auto_scan\Flake Photos v3'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Desired size
target_size = (1280, 960)

# Loop through all PNG images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path) as img:
                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(output_path, 'PNG')
                print(f"Resized: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

print("✅ All images resized.")
