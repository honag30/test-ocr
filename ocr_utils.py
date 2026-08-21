import cv2
import re

def segment_lines(image):
    """
    Phân đoạn dòng chữ (Line Segmentation) sử dụng Morphology & Contours
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = binary.shape
    kernel_len = max(20, w // 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_boxes = []
    for cnt in contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        if box_h >= 10 and box_w >= 20:
            line_boxes.append((x, y, box_w, box_h))
            
    line_boxes = sorted(line_boxes, key=lambda b: b[1])
    
    line_crops = []
    for (x, y, box_w, box_h) in line_boxes:
        pad = 4
        y1 = max(0, y - pad)
        y2 = min(h, y + box_h + pad)
        x1 = max(0, x - pad)
        x2 = min(w, x + box_w + pad)
        crop = image[y1:y2, x1:x2]
        line_crops.append(crop)
        
    return line_crops, line_boxes

def group_results_by_line(raw_results, y_tolerance=20):
    """
    Gom nhóm các cụm chữ nằm trên CÙNG MỘT DÒNG NGANG (tọa độ Y tương đương).
    Trả về tuple: (formatted_lines, line_confidences)
    """
    boxes = []
    for item in raw_results:
        bbox, text, prob = item[0], item[1], item[2]
        if prob < 0.15:
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        y_center = sum(ys) / len(ys)
        x_min = min(xs)
        boxes.append({'bbox': bbox, 'text': text, 'prob': prob, 'y_center': y_center, 'x_min': x_min})
    
    # Sắp xếp theo Y_center từ trên xuống dưới
    boxes.sort(key=lambda b: b['y_center'])
    
    lines = []
    current_line = []
    
    for box in boxes:
        if not current_line:
            current_line.append(box)
        else:
            if abs(box['y_center'] - current_line[0]['y_center']) <= y_tolerance:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    if current_line:
        lines.append(current_line)
        
    formatted_lines = []
    line_confidences = []
    
    for line in lines:
        line.sort(key=lambda b: b['x_min']) # Xếp từ trái sang phải
        line_text = " ".join([b['text'] for b in line])
        
        # Chuẩn hóa khoảng cách và dấu gạch nối giữa Thời gian (hh:mm - dd/mm/yyyy)
        line_text = re.sub(r'(\d{2}:\d{2})\s*[\~2\=]\s*(\d{2}/\d{2}/\d{4})', r'\1 - \2', line_text)
        
        # Tính độ tin cậy trung bình của dòng
        avg_prob = sum([b['prob'] for b in line]) / len(line) if line else 0.0
        
        formatted_lines.append(line_text)
        line_confidences.append(avg_prob)
        
    return formatted_lines, line_confidences
