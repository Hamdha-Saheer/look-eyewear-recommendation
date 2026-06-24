# balance_data.py
# Updated after removing 98 bad images
# Male   target: 128 per class (round is now smallest at 128)
# Female target: 623 per class (heart is now smallest at 623)
# Does NOT touch valid/ or test/ folders
# Does NOT touch data/ folder

import os, random

def balance_train(gender, target):
    base = f'data_clean/{gender}/train'
    print(f'\n{"="*45}')
    print(f' Balancing {gender}/train -> {target} per class')
    print(f'{"="*45}')

    for shape in sorted(os.listdir(base)):
        folder = os.path.join(base, shape)
        if not os.path.isdir(folder):
            continue

        imgs = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
        current = len(imgs)

        if current <= target:
            print(f'  {shape:10s}: {current:4d} -> keep all '
                  f'(already at or below {target})')
        else:
            random.seed(42)
            random.shuffle(imgs)
            to_remove = imgs[target:]
            for f in to_remove:
                os.remove(os.path.join(folder, f))
            print(f'  {shape:10s}: {current:4d} -> {target} '
                  f'(removed {len(to_remove)})')

    # Verify final counts
    print(f'\n  Verification - {gender}/train final counts:')
    total = 0
    for shape in sorted(os.listdir(base)):
        folder = os.path.join(base, shape)
        if os.path.isdir(folder):
            n = len([f for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
            bar = '█' * (n // 10)
            print(f'    {shape:10s}: {n:4d}  {bar}')
            total += n
    print(f'    {"TOTAL":10s}: {total}')

# ── Run ───────────────────────────────────────────────────────────────
# Male:   smallest class is round with 128 images
# Female: smallest class is heart with 623 images
balance_train('male',   128)
balance_train('female', 623)

print('\nDone! Both train folders are now balanced.')
print('Valid and test folders were NOT changed.')