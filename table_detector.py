import cv2
import numpy as np
import re
import os
import json

def detect_table_grid_cells(image):
    """
    Phát hiện các ô (Cell) của Bảng từ hình ảnh bằng OpenCV Morphological Operations.
    Trả về danh sách các ô: [{'x': x, 'y': y, 'w': cw, 'h': ch}, ...]
    và bounding box tổng của Bảng (x, y, w, h).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binarize (Nhị phân hóa)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    
    # 1. Trích xuất đường kẻ ngang (Horizontal lines)
    scale_h = max(20, w // 30)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_h, 1))
    horizontal = cv2.erode(binary, kernel_h, iterations=1)
    horizontal = cv2.dilate(horizontal, kernel_h, iterations=1)

    # 2. Trích xuất đường kẻ dọc (Vertical lines)
    scale_v = max(15, h // 40)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_v))
    vertical = cv2.erode(binary, kernel_v, iterations=1)
    vertical = cv2.dilate(vertical, kernel_v, iterations=1)

    # 3. Kết hợp đường kẻ ngang & dọc để tạo lưới Bảng (Table Grid)
    table_grid = cv2.add(horizontal, vertical)

    # Tìm các contour của ô lưới
    contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    min_cell_w, min_cell_h = 20, 12
    max_cell_w, max_cell_h = int(w * 0.95), int(h * 0.95)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if min_cell_w <= cw <= max_cell_w and min_cell_h <= ch <= max_cell_h:
            cells.append({'x': int(x), 'y': int(y), 'w': int(cw), 'h': int(ch)})

    if not cells:
        return [], None

    # Lọc trùng lặp ô gần nhau
    unique_cells = []
    cells.sort(key=lambda c: (c['y'], c['x']))
    for c in cells:
        is_duplicate = False
        for u in unique_cells:
            if abs(c['x'] - u['x']) < 8 and abs(c['y'] - u['y']) < 8 and abs(c['w'] - u['w']) < 12 and abs(c['h'] - u['h']) < 12:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_cells.append(c)

    if not unique_cells:
        return [], None

    min_x = min(c['x'] for c in unique_cells)
    min_y = min(c['y'] for c in unique_cells)
    max_r = max(c['x'] + c['w'] for c in unique_cells)
    max_b = max(c['y'] + c['h'] for c in unique_cells)
    table_bbox = [int(min_x), int(min_y), int(max_r - min_x), int(max_b - min_y)]

    return unique_cells, table_bbox


def extract_table_from_raw_results(raw_results, image=None, y_tolerance=18, x_tolerance=25):
    """
    Trích xuất Bảng và thông tin vị trí Bounding Box chi tiết từ EasyOCR raw_results.
    Trả về:
      - table_matrix: List[List[str]]
      - is_table_found: bool
      - structured_table: dict chứa thông tin ô, vị trí bbox và các đoạn chữ
    """
    if not raw_results:
        return [], False, {}

    ocr_boxes = []
    for item in raw_results:
        bbox, text, prob = item[0], item[1].strip(), item[2]
        if prob < 0.15 or not text:
            continue
        # Format bbox -> list of float/int
        clean_bbox = [[float(pt[0]), float(pt[1])] for pt in bbox]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        x_min, x_max = float(min(xs)), float(max(xs))
        y_min, y_max = float(min(ys)), float(max(ys))
        y_center = (y_min + y_max) / 2.0
        x_center = (x_min + x_max) / 2.0
        ocr_boxes.append({
            'bbox': clean_bbox,
            'text': text,
            'prob': float(prob),
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'x_center': x_center,
            'y_center': y_center,
            'width': x_max - x_min,
            'height': y_max - y_min
        })

    if not ocr_boxes:
        return [], False, {}

    # TH1: OpenCV Cell Detection cho ảnh
    grid_cells, table_bbox = None, None
    if image is not None:
        try:
            grid_cells, table_bbox = detect_table_grid_cells(image)
        except Exception:
            grid_cells = None

    if grid_cells and len(grid_cells) >= 4:
        grid_cells.sort(key=lambda c: c['y'])
        rows = []
        for cell in grid_cells:
            assigned = False
            for r in rows:
                if abs(cell['y'] - r[0]['y']) <= y_tolerance:
                    r.append(cell)
                    assigned = True
                    break
            if not assigned:
                rows.append([cell])

        valid_rows = [r for r in rows if len(r) >= 2]
        if len(valid_rows) >= 2:
            for r in valid_rows:
                r.sort(key=lambda c: c['x'])

            all_cols_x = []
            for r in valid_rows:
                for c in r:
                    all_cols_x.append(c['x'])
            all_cols_x.sort()

            col_clusters = []
            for x in all_cols_x:
                if not col_clusters:
                    col_clusters.append([x])
                else:
                    if abs(x - sum(col_clusters[-1])/len(col_clusters[-1])) <= x_tolerance:
                        col_clusters[-1].append(x)
                    else:
                        col_clusters.append([x])

            num_cols = len(col_clusters)
            if num_cols >= 2:
                matrix = [["" for _ in range(num_cols)] for _ in range(len(valid_rows))]
                cell_details = []

                def get_col_index(cell_x):
                    best_idx = 0
                    min_dist = float('inf')
                    for idx, cl in enumerate(col_clusters):
                        mean_x = sum(cl) / len(cl)
                        dist = abs(cell_x - mean_x)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx
                    return best_idx

                for r_idx, r in enumerate(valid_rows):
                    for c in r:
                        col_idx = get_col_index(c['x'])
                        cell_texts = []
                        cell_items = []
                        for box in ocr_boxes:
                            bx, by = box['x_center'], box['y_center']
                            if c['x'] - 5 <= bx <= c['x'] + c['w'] + 5 and c['y'] - 5 <= by <= c['y'] + c['h'] + 5:
                                cell_texts.append(box['text'])
                                cell_items.append({
                                    'text': box['text'],
                                    'confidence': round(box['prob'], 4),
                                    'bbox': box['bbox']
                                })

                        combined_text = " ".join(cell_texts)
                        if combined_text:
                            matrix[r_idx][col_idx] = (matrix[r_idx][col_idx] + " " + combined_text).strip()
                            cell_details.append({
                                'row_index': r_idx,
                                'col_index': col_idx,
                                'cell_bbox': [c['x'], c['y'], c['w'], c['h']],
                                'text': matrix[r_idx][col_idx],
                                'text_segments': cell_items
                            })

                filtered_matrix = [r for r in matrix if any(cell.strip() for cell in r)]
                if len(filtered_matrix) >= 2:
                    structured_table = {
                        'detection_method': 'opencv_grid',
                        'table_bbox': table_bbox,
                        'num_rows': len(filtered_matrix),
                        'num_cols': num_cols,
                        'matrix': filtered_matrix,
                        'cells': cell_details
                    }
                    return filtered_matrix, True, structured_table

    # TH2: Spatial Reconstruction dựa trên vị trí cột
    ocr_boxes.sort(key=lambda b: b['y_center'])
    lines = []
    for box in ocr_boxes:
        if not lines:
            lines.append([box])
        else:
            if abs(box['y_center'] - lines[-1][0]['y_center']) <= y_tolerance:
                lines[-1].append(box)
            else:
                lines.append([box])

    table_candidate_lines = [l for l in lines if len(l) >= 3 or (len(l) >= 2 and any(re.search(r'\d{3,}', b['text']) for b in l))]

    if len(table_candidate_lines) < 2:
        return [], False, {}

    xs_all = []
    for l in table_candidate_lines:
        for b in l:
            xs_all.append(b['x_min'])
    xs_all.sort()

    cols = []
    for x in xs_all:
        if not cols:
            cols.append([x])
        else:
            avg_x = sum(cols[-1]) / len(cols[-1])
            if abs(x - avg_x) <= 60:
                cols[-1].append(x)
            else:
                cols.append([x])

    if len(cols) < 2:
        return [], False, {}

    col_bounds = []
    for c in cols:
        col_bounds.append((min(c) - 15, max(c) + 60))

    matrix = []
    cell_details = []

    for r_idx, l in enumerate(lines):
        row = [""] * len(col_bounds)
        l.sort(key=lambda b: b['x_min'])
        has_content = False
        for b in l:
            best_col = -1
            min_diff = float('inf')
            for c_idx, (c_min, c_max) in enumerate(col_bounds):
                diff = abs(b['x_min'] - ((c_min + c_max) / 2.0))
                if diff < min_diff:
                    min_diff = diff
                    best_col = c_idx

            if best_col != -1:
                if row[best_col]:
                    row[best_col] += " " + b['text']
                else:
                    row[best_col] = b['text']
                has_content = True

                cell_details.append({
                    'row_index': r_idx,
                    'col_index': best_col,
                    'cell_bbox': [round(b['x_min'], 1), round(b['y_min'], 1), round(b['width'], 1), round(b['height'], 1)],
                    'text': b['text'],
                    'text_segments': [{
                        'text': b['text'],
                        'confidence': round(b['prob'], 4),
                        'bbox': b['bbox']
                    }]
                })

        if has_content:
            matrix.append(row)

    if len(matrix) >= 2:
        structured_table = {
            'detection_method': 'spatial_clustering',
            'num_rows': len(matrix),
            'num_cols': len(col_bounds),
            'matrix': matrix,
            'cells': cell_details
        }
        return matrix, True, structured_table

    return [], False, {}


def extract_tables_from_pdf_digital(pdf_path):
    """
    Trích xuất bảng & tọa độ bbox từng ô từ file PDF kỹ thuật số bằng pdfplumber.
    Trả về tuple: (tables_matrices, structured_tables)
    """
    tables_found = []
    structured_tables = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, 1):
                # Lấy danh sách từ kèm vị trí bbox
                words = page.extract_words()
                extracted_tables = page.find_tables()

                for t_idx, table_obj in enumerate(extracted_tables):
                    extracted = table_obj.extract()
                    if not extracted:
                        continue

                    cleaned_table = []
                    cell_details = []

                    for r_idx, row in enumerate(extracted):
                        cleaned_row = [cell.strip().replace("\n", " ") if cell else "" for cell in row]
                        if any(cleaned_row):
                            cleaned_table.append(cleaned_row)

                    if len(cleaned_table) >= 2:
                        tables_found.append(cleaned_table)

                        # Trích xuất chi tiết ô + bbox từng ô từ pdfplumber
                        for r_idx, row_cells in enumerate(table_obj.rows):
                            for c_idx, cell_bbox in enumerate(row_cells.cells):
                                if cell_bbox is None:
                                    continue
                                x0, top, x1, bottom = cell_bbox
                                cell_words = [
                                    {
                                        'text': w['text'],
                                        'bbox': [round(w['x0'], 2), round(w['top'], 2), round(w['x1'], 2), round(w['bottom'], 2)]
                                    }
                                    for w in words
                                    if x0 - 2 <= w['x0'] and w['x1'] <= x1 + 2 and top - 2 <= w['top'] and w['bottom'] <= bottom + 2
                                ]

                                text_in_cell = " ".join(w['text'] for w in cell_words)
                                cell_details.append({
                                    'row_index': r_idx,
                                    'col_index': c_idx,
                                    'cell_bbox': [round(x0, 2), round(top, 2), round(x1 - x0, 2), round(bottom - top, 2)],
                                    'text': text_in_cell,
                                    'text_segments': cell_words
                                })

                        structured_tables.append({
                            'page': page_idx,
                            'detection_method': 'pdfplumber_digital',
                            'num_rows': len(cleaned_table),
                            'num_cols': max(len(r) for r in cleaned_table),
                            'matrix': cleaned_table,
                            'cells': cell_details
                        })
    except Exception as e:
        print(f"pdfplumber không khả dụng hoặc lỗi: {e}")
    return tables_found, structured_tables


def render_pdf_to_images(pdf_path):
    """
    Chuyển các trang PDF thành ảnh OpenCV BGR sử dụng PyMuPDF (fitz).
    """
    images = []
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            images.append(img_np)
    except Exception as e:
        print(f"PyMuPDF/fitz không khả dụng hoặc lỗi: {e}")
    return images


def format_table_to_markdown(table_matrix):
    """
    Chuyển ma trận Bảng 2D (List[List[str]]) thành chuỗi Markdown Table hoàn chỉnh.
    """
    if not table_matrix:
        return ""

    num_cols = max(len(row) for row in table_matrix)
    norm_matrix = []
    for row in table_matrix:
        r = [cell.strip().replace("\n", " ") for cell in row]
        if len(r) < num_cols:
            r.extend([""] * (num_cols - len(r)))
        norm_matrix.append(r)

    col_widths = [3] * num_cols
    for row in norm_matrix:
        for c_idx, cell in enumerate(row):
            col_widths[c_idx] = max(col_widths[c_idx], len(cell))

    header = norm_matrix[0]
    header_line = "| " + " | ".join(header[i].ljust(col_widths[i]) for i in range(num_cols)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |"

    body_lines = []
    for row in norm_matrix[1:]:
        line = "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(num_cols)) + " |"
        body_lines.append(line)

    md_table = [header_line, sep_line] + body_lines
    return "\n".join(md_table)
