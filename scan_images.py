# scan_images.py
# Run this: python scan_images.py
# SAFE — nothing is deleted, only reports bad images

import os, cv2
import mediapipe as mp

data     = 'data_clean'
MIN_CONF = 0.6
total    = 0
bad      = 0
bad_list = []

mp_detect = mp.solutions.face_detection

print('Scanning for bad images...')
print('This will take 5-10 minutes for 5957 images. Please wait...')
print()

with mp_detect.FaceDetection(
        model_selection=1,
        min_detection_confidence=MIN_CONF) as det:

    for gender in sorted(os.listdir(data)):
        for split in ['train', 'valid', 'test']:
            for shape in ['heart','oblong','oval','round','square']:
                folder = os.path.join(data, gender, split, shape)
                if not os.path.exists(folder):
                    continue

                imgs = [f for f in os.listdir(folder)
                        if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]

                bad_here = 0

                for img_name in imgs:
                    path = os.path.join(folder, img_name)
                    img  = cv2.imread(path)

                    if img is None:
                        bad_here += 1
                        bad      += 1
                        bad_list.append((path, 'Cannot open file'))
                        total    += 1
                        continue

                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = det.process(rgb)

                    if not res.detections:
                        bad_here += 1
                        bad      += 1
                        bad_list.append((path, 'No face detected'))
                    else:
                        score = res.detections[0].score[0]
                        if score < MIN_CONF:
                            bad_here += 1
                            bad      += 1
                            bad_list.append((path,
                                f'Low confidence {score:.0%}'))

                    total += 1

                status = 'OK ' if bad_here == 0 else 'BAD'
                print(f'  [{status}] {gender:6s} {split:5s} '
                      f'{shape:10s} — bad: {bad_here}')

# Show bad image list
if bad_list:
    print()
    print('='*50)
    print(f'  Bad images (first 30 shown):')
    print('='*50)
    for path, reason in bad_list[:30]:
        short = path.replace('data_clean' + os.sep, '')
        print(f'  {short}')
        print(f'  -> {reason}')
        print()
    if len(bad_list) > 30:
        print(f'  ... and {len(bad_list)-30} more not shown')

# Final summary
print()
print('='*50)
print('  FINAL SUMMARY')
print('='*50)
print(f'  Total scanned : {total}')
print(f'  Good images   : {total - bad}')
print(f'  Bad images    : {bad}')
if total > 0:
    print(f'  Bad rate      : {bad/total:.1%}')
print()

if bad == 0:
    print('  DECISION: All images are clean')
    print('            Submit your project as is')
elif bad < 20:
    print(f'  DECISION: Only {bad} bad images found')
    print('            Very small improvement expected')
    print('            Skip cleaning — submit now')
elif bad < 50:
    print(f'  DECISION: {bad} bad images found')
    print('            Worth cleaning if you have 3+ hours')
    print('            Set DRY_RUN=False in remove_bad_images.py')
    print('            Then retrain both models')
else:
    print(f'  DECISION: {bad} bad images found')
    print('            Definitely clean before retraining')
    print('            Set DRY_RUN=False in remove_bad_images.py')
    print('            Then run balance_data.py and retrain')

print()
print('  SAFE MODE — nothing was deleted')
print('='*50)