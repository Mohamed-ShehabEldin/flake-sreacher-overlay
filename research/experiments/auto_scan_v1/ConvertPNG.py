import os
from PIL import Image

def convert_jpg_to_png(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print("The specified folder does not exist.")
        return

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            jpg_path = os.path.join(folder_path, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(folder_path, png_filename)

            try:
                # Open the JPG image
                with Image.open(jpg_path) as img:
                    # Convert and save as PNG
                    img.save(png_path, "PNG")
                print(f"Converted: {filename} → {png_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    folder = r"C:\Users\QMLab\Desktop\auto_scan\FLAKES"
    convert_jpg_to_png(folder)
