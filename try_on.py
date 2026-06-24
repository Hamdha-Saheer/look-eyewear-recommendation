
import cv2, numpy as np, os

def overlay_glasses(face_img_path, img_filename, landmarks, output_path):
    """
    Overlay a specific glasses PNG onto a face photo.
    img_filename: e.g. 'wayfarer_2.png'  (file in static/glasses/)
    """
    gpath = f'static/glasses/{img_filename}'
    if not os.path.exists(gpath):
        print(f'Glasses file not found: {gpath}')
        return False

    face    = cv2.imread(face_img_path)
    glasses = cv2.imread(gpath, cv2.IMREAD_UNCHANGED)
    if face is None or glasses is None:
        return False

    # Ensure glasses has alpha channel
    if glasses.shape[2] == 3:
        glasses = cv2.cvtColor(glasses, cv2.COLOR_BGR2BGRA)

    lt = landmarks['left_temple']
    rt = landmarks['right_temple']
    nb = landmarks['nose_bridge']

    # Face tilt angle
    dx    = rt[0] - lt[0]
    dy    = rt[1] - lt[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Scale to inter-temple distance
    face_w = int(np.linalg.norm(np.array(rt) - np.array(lt)) * 1.15)
    aspect = glasses.shape[0] / glasses.shape[1]
    face_h = int(face_w * aspect)

    g  = cv2.resize(glasses, (face_w, face_h))
    cx, cy = face_w // 2, face_h // 2
    M  = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    gr = cv2.warpAffine(g, M, (face_w, face_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0))

    # Position centred on nose bridge
    x1 = nb[0] - face_w // 2
    y1 = nb[1] - face_h // 2
    ih, iw = face.shape[:2]
    fx1 = max(0, x1);      fy1 = max(0, y1)
    fx2 = min(iw, x1+face_w); fy2 = min(ih, y1+face_h)
    gx1 = fx1 - x1;       gy1 = fy1 - y1
    gx2 = gx1 + (fx2-fx1); gy2 = gy1 + (fy2-fy1)

    # Alpha blend
    alpha    = gr[gy1:gy2, gx1:gx2, 3:4] / 255.0
    face_roi = face[fy1:fy2, fx1:fx2]
    for c in range(3):
        face_roi[:,:,c] = ((1 - alpha[:,:,0]) * face_roi[:,:,c] +
                            alpha[:,:,0]       * gr[gy1:gy2, gx1:gx2, c])
    face[fy1:fy2, fx1:fx2] = face_roi
    cv2.imwrite(output_path, face)
    return True