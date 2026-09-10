import cv2
import numpy as np

# Ngưỡng kích thước tối thiểu để cân nhắc upscale (pixel)
_MIN_UPSCALE_DIM = 1200

def deskew_image(image):
    """
    Tự động phát hiện góc nghiêng và xoay thẳng ảnh (Deskewing)
    """
    coords = np.column_stack(np.where(image < 255))
    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def remove_background(gray, fast_mode: bool = False):
    """
    Tự động tách/làm sạch nền bằng hình thái học (Morphological Background Removal).
    fast_mode: dùng kernel nhỏ hơn để tăng tốc.
    """
    ksize = (15, 15) if fast_mode else (21, 21)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg)
    norm = cv2.divide(gray, bg, scale=255)
    return norm

def remove_shadows(gray, fast_mode: bool = False):
    """
    Tự động loại bỏ bóng râm / bóng đổ không đều (Shadow Removal).
    fast_mode: dùng median blur nhỏ hơn để tăng tốc (~4x nhanh hơn).
    """
    dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    blur_size = 11 if fast_mode else 21
    bg_shadow = cv2.medianBlur(dilated, blur_size)
    diff = 255 - cv2.absdiff(gray, bg_shadow)
    normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return normalized

def enhance_edges(image):
    """
    Tăng cường đường biên nét chữ (Edge Enhancement)
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    enhanced = cv2.addWeighted(image, 1.0, laplacian_abs, -0.3, 0)
    return enhanced

def apply_morph_closing(image, kernel_size=(2, 2)):
    """
    Phép đóng hình thái học (Morphological Closing)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed

def preprocess_image(
    image_input,
    sharpen=True,
    denoise=True,
    threshold=False,
    enhance_contrast=True,
    deskew=True,
    bg_removal=True,
    shadow_removal=True,
    edge_enhance=False,
    morph_close=False,
    image_input_is_np=False,
    fast_mode=True,
):
    """
    Tiền xử lý hình ảnh tối ưu giữ nguyên nét chữ và dấu tiếng Việt.
    Hỗ trợ đầu vào là đường dẫn file (str) hoặc Mảng NumPy OpenCV (np.ndarray).

    fast_mode=True (mặc định):
      - Chỉ upscale nếu ảnh quá nhỏ (< 1200px theo chiều ngắn)
      - Dùng GaussianBlur thay fastNlMeansDenoising (nhanh hơn ~10-30x)
      - Dùng kernel nhỏ hơn cho shadow/bg removal
    fast_mode=False: giữ hành vi cũ, chất lượng tối đa (chậm hơn nhiều).
    """
    if isinstance(image_input, np.ndarray) or image_input_is_np:
        img = image_input.copy()
    else:
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {image_input}")

    # 1. Upscale thông minh: chỉ phóng to nếu ảnh quá nhỏ
    height, width = img.shape[:2]
    min_dim = min(height, width)
    if min_dim < _MIN_UPSCALE_DIM:
        # Upscale vừa đủ để đạt ngưỡng tối thiểu (tránh phóng quá to)
        scale = min(2.0, _MIN_UPSCALE_DIM / min_dim)
        new_w = int(width * scale)
        new_h = int(height * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Nếu ảnh đủ lớn (từ PDF DPI cao hay ảnh gốc lớn) → không upscale

    # 2. Chuyển sang ảnh xám (Grayscale)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 3. Triệt tiêu bóng râm
    if shadow_removal:
        gray = remove_shadows(gray, fast_mode=fast_mode)

    # 4. Tách / Khử nền ảnh
    if bg_removal:
        gray = remove_background(gray, fast_mode=fast_mode)

    # 5. Xoay thẳng ảnh nghiêng
    if deskew:
        gray = deskew_image(gray)

    # 6. Tăng tương phản nhẹ nhàng
    if enhance_contrast:
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)

    # 7. Khử nhiễu
    if denoise:
        if fast_mode:
            # GaussianBlur: nhanh hơn fastNlMeansDenoising ~10-30x
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        else:
            # Chất lượng cao hơn nhưng chậm hơn đáng kể
            gray = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)

    # 8. CLAHE — tăng cường tương phản cục bộ
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 9. Biên nét chữ (tùy chọn)
    if edge_enhance:
        enhanced = enhance_edges(enhanced)

    # 10. Phân ngưỡng nhị phân (nếu bật)
    if threshold:
        _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 11. Phép đóng hình thái học (tùy chọn)
    if morph_close:
        enhanced = apply_morph_closing(enhanced)

    # 12. Làm nét ảnh Unsharp Masking
    if sharpen:
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)

    return enhanced
