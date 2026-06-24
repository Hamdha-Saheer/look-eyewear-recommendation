import os
import cv2
from face_utils import crop_face

# Set this to 'female' first, then run it again for 'male'
GENDER = 'male' 

def clean_folder(subfolder):
    raw_path = f'data/{GENDER}/{subfolder}'
    clean_path = f'data_clean/{GENDER}/{subfolder}'
    
    if not os.path.exists(raw_path):
        return

    for category in os.listdir(raw_path):
        os.makedirs(os.path.join(clean_path, category), exist_ok=True)
        print(f"Cleaning {subfolder}/{category}...")
        
        for img_name in os.listdir(os.path.join(raw_path, category)):
            img_path = os.path.join(raw_path, category, img_name)
            save_path = os.path.join(clean_path, category, img_name)
            
            # Skip if already exists
            if os.path.exists(save_path): continue

            # USE YOUR FACE_UTILS TO CROP
            face = crop_face(img_path)
            
            if face is not None:
                cv2.imwrite(save_path, face)

# Clean all three splits
clean_folder('train')
clean_folder('valid')
clean_folder('test')
print(" Done! Your high-quality dataset is in 'data_clean'")