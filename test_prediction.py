from PIL import Image
import os

# This script checks if your glasses have a transparent background
folder_path = 'static/glasses'

if not os.path.exists(folder_path):
    print(f"ERROR: The folder '{folder_path}' does not exist. Create it first!")
else:
    for f in os.listdir(folder_path):
        if not f.endswith('.png'): 
            continue
            
        img = Image.open(os.path.join(folder_path, f))
        
        # 'RGBA' means the image has an Alpha (transparency) channel
        if img.mode != 'RGBA':
            print(f'ERROR {f}: No alpha - mode={img.mode} | FIX: re-download as PNG-32')
        else:
            print(f'OK    {f}: Transparent background confirmed')