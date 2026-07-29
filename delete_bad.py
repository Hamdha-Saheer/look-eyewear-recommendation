# delete_bad.py
# Run: python delete_bad.py
# This will DELETE bad images from data_clean/

import os
import sys
import cv2
import mediapipe as mp

DATA_FOLDER    = 'data_clean'
MIN_CONFIDENCE = 0.6

mp_detect = mp.solutions.face_detection

deleted = 0
kept    = 0

sys.stdout.write('Starting deletion of bad images...\n')
sys.stdout.flush()

with mp_detect.FaceDetection(
        model_selection=1,
        min_detection_confidence=MIN_CONFIDENCE) as det:

    for gender in sorted(os.listdir(DATA_FOLDER)):
        gpath = os.path.join(DATA_FOLDER, gender)
        if not os.path.isdir(gpath):
            continue

        for split in ['train', 'valid', 'test']:
            spath = os.path.join(gpath, split)
            if not os.path.exists(spath):
                continue

            for shape in ['heart', 'oblong', 'oval', 'round', 'square']:
                fpath = os.path.join(spath, shape)
                if not os.path.exists(fpath):
                    continue

                imgs = [f for f in os.listdir(fpath)
                        if f.lower().endswith(
                            ('.jpg', '.jpeg', '.png', '.webp'))]

                bad_here = 0

                for img_name in imgs:
                    img_path = os.path.join(fpath, img_name)

                    # Try to open
                    img = cv2.imread(img_path)
                    if img is None:
                        os.remove(img_path)
                        deleted += 1
                        bad_here += 1
                        continue

                    # Check face
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = det.process(rgb)

                    if not res.detections:
                        os.remove(img_path)
                        deleted += 1
                        bad_here += 1
                    else:
                        score = res.detections[0].score[0]
                        if score < MIN_CONFIDENCE:
                            os.remove(img_path)
                            deleted += 1
                            bad_here += 1
                        else:
                            kept += 1

                line = (f'  {gender:6s} {split:5s} {shape:10s}'
                        f' — deleted: {bad_here}\n')
                sys.stdout.write(line)
                sys.stdout.flush()

sys.stdout.write('\n')
sys.stdout.write('=' * 45 + '\n')
sys.stdout.write(f'  DONE\n')
sys.stdout.write(f'  Deleted : {deleted} bad images\n')
sys.stdout.write(f'  Kept    : {kept} good images\n')
sys.stdout.write('\n')
sys.stdout.write('  Next steps:\n')
sys.stdout.write('    1. python balance_data.py\n')
sys.stdout.write('    2. python train_model.py --gender male\n')
sys.stdout.write('    3. python train_model.py --gender female\n')
sys.stdout.write('=' * 45 + '\n')
sys.stdout.flush()