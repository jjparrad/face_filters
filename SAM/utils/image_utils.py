from PIL import Image
import numpy as np
import cv2
import os


def preprocess_image(input_image, opts=None, method='face_crop'):


    size = getattr(opts, 'preprocess_size', 256) if opts is not None else 256
    # Convert PIL -> OpenCV BGR
    img = cv2.cvtColor(np.array(input_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haar cascade classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_classifier.detectMultiScale(
        grey_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # If no faces found, return resized original
    if len(faces) == 0:
        return input_image.resize((size, size))

    best_area = 0
    best_crop = None
    for (x, y, w, h) in faces:
        if w <= 0 or h <= 0:
            continue
        x_max = min(img.shape[1], x + w)
        y_max = min(img.shape[0], y + h)
        x0 = max(0, x)
        y0 = max(0, y)
        im_cropped = img[y0:y_max, x0:x_max]
        if im_cropped.size == 0:
            continue
        area = (x_max - x0) * (y_max - y0)
        if area > best_area:
            best_area = area
            best_crop = im_cropped

    if best_crop is None:
        return input_image.resize((size, size))

    # back to RGB and resize
    best_crop = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
    best_crop = cv2.resize(best_crop, (size, size))

    # optionally save debug crop inside exp_dir
    try:
        if opts is not None and hasattr(opts, 'exp_dir') and opts.exp_dir:
            debug_dir = os.path.join(opts.exp_dir, 'preprocess_debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, 'crop.png')
            
            # If file exists, increment counter
            counter = 2
            base_path = debug_path.replace('.png', '')
            while os.path.exists(debug_path):
                debug_path = f"{base_path}_{counter}.png"
                counter += 1
            
            cv2.imwrite(debug_path, cv2.cvtColor(best_crop, cv2.COLOR_RGB2BGR))
    except Exception:
        # Don't fail inference for debug writing errors
        pass

    best_crop = best_crop.astype(np.uint8)
    pil_img = Image.fromarray(best_crop)
    return pil_img

