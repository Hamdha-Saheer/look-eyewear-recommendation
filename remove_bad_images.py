# remove_bad_images.py
# DRY_RUN = True means NOTHING is deleted — only shows what would be removed
# Change DRY_RUN = False only when you are ready to actually delete

import os
import cv2
import mediapipe as mp

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_FOLDER    = 'data_clean'
MIN_CONFIDENCE = 0.6
DRY_RUN        = False   # SAFE MODE — nothing deleted until you change this

# ── Setup ─────────────────────────────────────────────────────────────
mp_detect = mp.solutions.face_detection

def check_image(img_path, detector):
    img = cv2.imread(img_path)
    if img is None:
        return False, 'Cannot open file'

    h, w = img.shape[:2]
    if h < 50 or w < 50:
        return False, f'Too small ({w}x{h})'

    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if not result.detections:
        return False, 'No face detected'

    best_score = max(d.score[0] for d in result.detections)
    if best_score < MIN_CONFIDENCE:
        return False, f'Low confidence ({best_score:.0%})'

    return True, f'OK ({best_score:.0%})'

def scan_and_clean():
    total_checked = 0
    total_bad     = 0
    total_good    = 0
    bad_images    = []

    print(f'\n{"="*55}')
    print(f'  Scanning: {DATA_FOLDER}/')
    print(f'  Mode:     {"DRY RUN - nothing will be deleted" if DRY_RUN else "LIVE - bad images WILL be deleted"}')
    print(f'  Min face confidence: {MIN_CONFIDENCE:.0%}')
    print(f'{"="*55}\n')

    with mp_detect.FaceDetection(
            model_selection=1,
            min_detection_confidence=MIN_CONFIDENCE) as detector:

        for gender in sorted(os.listdir(DATA_FOLDER)):
            gender_path = os.path.join(DATA_FOLDER, gender)
            if not os.path.isdir(gender_path):
                continue

            for split in ['train', 'valid', 'test']:
                split_path = os.path.join(gender_path, split)
                if not os.path.exists(split_path):
                    continue

                for shape in sorted(os.listdir(split_path)):
                    shape_path = os.path.join(split_path, shape)
                    if not os.path.isdir(shape_path):
                        continue

                    imgs = [f for f in os.listdir(shape_path)
                            if f.lower().endswith(
                                ('.jpg','.jpeg','.png','.webp'))]

                    bad_in_folder  = 0
                    good_in_folder = 0

                    for img_name in imgs:
                        img_path = os.path.join(shape_path, img_name)
                        is_good, reason = check_image(img_path, detector)
                        total_checked  += 1

                        if is_good:
                            good_in_folder += 1
                            total_good     += 1
                        else:
                            bad_in_folder += 1
                            total_bad     += 1
                            bad_images.append((img_path, reason))

                            if not DRY_RUN:
                                os.remove(img_path)

                    icon = 'OK' if bad_in_folder == 0 else 'BAD'
                    print(f'  [{icon}] {gender:6s} {split:5s} {shape:10s}'
                          f' — good:{good_in_folder:4d}  bad:{bad_in_folder:3d}')

    # Print bad image details
    if bad_images:
        print(f'\n{"="*55}')
        print(f'  Bad images found: {len(bad_images)}')
        print(f'{"="*55}')
        for path, reason in bad_images[:30]:
            action = 'WOULD DELETE' if DRY_RUN else 'DELETED'
            short  = path.replace(DATA_FOLDER + os.sep, '')
            print(f'  {action}: {short}')
            print(f'           Reason : {reason}')
        if len(bad_images) > 30:
            print(f'  ... and {len(bad_images)-30} more (not shown)')
    else:
        print(f'\n  All images are clean - no bad images found!')

    # Summary
    print(f'\n{"="*55}')
    print(f'  SUMMARY')
    print(f'{"="*55}')
    print(f'  Total scanned : {total_checked}')
    print(f'  Good images   : {total_good}')
    print(f'  Bad images    : {total_bad}')
    if total_checked > 0:
        print(f'  Bad rate      : {total_bad/total_checked:.1%}')

    print()
    if DRY_RUN:
        print(f'  *** DRY RUN - NOTHING WAS DELETED ***')
        print()
        if total_bad == 0:
            print(f'  DECISION: All clean - no action needed - submit as is')
        elif total_bad < 20:
            print(f'  DECISION: Only {total_bad} bad images - small improvement'
                  f' - optional to clean')
        elif total_bad < 50:
            print(f'  DECISION: {total_bad} bad images - worth cleaning'
                  f' if you have 3+ hours')
        else:
            print(f'  DECISION: {total_bad} bad images - definitely clean'
                  f' before retraining')
        print()
        print(f'  To actually delete bad images:')
        print(f'    1. Open remove_bad_images.py')
        print(f'    2. Change:  DRY_RUN = True')
        print(f'       To:      DRY_RUN = False')
        print(f'    3. Run:     python remove_bad_images.py')
    else:
        print(f'  {total_bad} bad images deleted from data_clean/')
        print(f'  Your originals in data/ are untouched')
        print(f'\n  Next steps:')
        print(f'    1. python balance_data.py')
        print(f'    2. python train_model.py --gender male')
        print(f'    3. python train_model.py --gender female')

if __name__ == '__main__':
    scan_and_clean()