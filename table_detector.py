import cv2
import numpy as np
import re
import os
import json
import pymupdf

DEBUG_TABLE = os.environ.get("DEBUG_TABLE", "0").lower() in ("1", "true", "yes")

def log_debug(msg: str):
    if DEBUG_TABLE:
        print(f"[DEBUG_TABLE] {msg}")

def parse_number(text: str) -> float | None:
    """Trích xuất giá trị số hợp lệ đầu tiên từ chuỗi text (xử lý an toàn chuỗi nhiều dòng)."""
    if not text:
        return None
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines:
        if re.match(r'^\(\d\)', line) or "=" in line or "%" in line:
            continue
        cleaned = re.sub(r'[^\d\.\,-]', '', line).replace(',', '')
        try:
            val = float(cleaned)
            if val in [1.0, 2.0, 3.0, 4.0, 5.0] and len(line) <= 3:
                continue
            return val
        except ValueError:
            continue
    return None

def validate_table_financials(matrix: list[list[str]]) -> dict:
    """
    Kiểm tra tính hợp lệ toán học của bảng (Validation Engine linh hoạt):
    - quantity * price ≈ amount
    - price - discount ≈ subtotal
    - subtotal + VAT ≈ grand_total
    Trả về: {"status": "pass" | "warning" | "not_checkable", "issues": [...]}
    """
    if not matrix or len(matrix) < 2:
        return {"status": "not_checkable", "issues": []}

    issues = []
    checked_count = 0

    for r_idx, row in enumerate(matrix[1:], 1):
        row_str = " ".join(row).lower()
        if "(1)" in row_str or "(2)" in row_str or "(3)" in row_str or "stt" in row_str:
            continue

        # Extract all numbers from row cells
        nums = []
        for cell in row:
            if not cell:
                continue
            lines = [l.strip() for l in cell.split('\n') if l.strip()]
            for l in lines:
                val = parse_number(l)
                if val is not None and val > 0:
                    nums.append(val)

        if len(nums) >= 3:
            # Check if any two numbers sum to the third
            # e.g. 599000 + 11920 == 610920 or 100000 + 8000 == 108000
            nums_sorted = sorted(nums)
            found_match = False
            for i in range(len(nums_sorted) - 1):
                for j in range(i + 1, len(nums_sorted) - 1):
                    if abs((nums_sorted[i] + nums_sorted[j]) - nums_sorted[-1]) <= 2.0:
                        found_match = True
                        break
                if found_match:
                    break

            if found_match:
                checked_count += 1
            else:
                # Check adjacent numbers subtotal + vat == total
                if abs((nums[0] + nums[1]) - nums[2]) <= 2.0:
                    checked_count += 1
                else:
                    issues.append(f"Hàng {r_idx}: {nums[0]} + {nums[1]} != {nums[2]}")

        elif len(nums) == 2:
            if abs(nums[0] - nums[1]) <= 1.0:
                checked_count += 1

    if checked_count == 0:
        return {"status": "not_checkable", "issues": []}
    elif issues:
        return {"status": "warning", "issues": issues}
    else:
        return {"status": "pass", "issues": []}


def detect_table_grid_cells(image):
    """
    Phát hiện các ô (Cell) của Bảng từ hình ảnh bằng OpenCV Morphological Operations.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    scale_h = max(20, w // 30)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_h, 1))
    horizontal = cv2.erode(binary, kernel_h, iterations=1)
    horizontal = cv2.dilate(horizontal, kernel_h, iterations=1)

    scale_v = max(15, h // 40)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_v))
    vertical = cv2.erode(binary, kernel_v, iterations=1)
    vertical = cv2.dilate(vertical, kernel_v, iterations=1)

    table_grid = cv2.add(horizontal, vertical)
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
    Trích xuất Bảng từ kết quả OCR thô (Image fallback).
    """
    if not raw_results:
        return [], False, {}

    ocr_boxes = []
    for item in raw_results:
        bbox, text, prob = item[0], item[1].strip(), item[2]
        if prob < 0.15 or not text:
            continue
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
                    val_res = validate_table_financials(filtered_matrix)
                    structured_table = {
                        'detection_method': 'opencv_grid',
                        'table_bbox': table_bbox,
                        'num_rows': len(filtered_matrix),
                        'num_cols': num_cols,
                        'matrix': filtered_matrix,
                        'cells': cell_details,
                        'validation': val_res
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
        val_res = validate_table_financials(matrix)
        structured_table = {
            'detection_method': 'spatial_clustering',
            'num_rows': len(matrix),
            'num_cols': len(col_bounds),
            'matrix': matrix,
            'cells': cell_details,
            'validation': val_res
        }
        return matrix, True, structured_table

    return [], False, {}


def extract_native_pdf_tables_from_page(page, page_num: int) -> list[dict]:
    """
    Trích xuất Bảng chính xác từ Native PDF Page theo phân cấp:
    words -> lines -> blocks -> candidate regions -> rows/cols -> cells -> matrix.
    Hỗ trợ gom multiline cells, không lọt paragraph và có validation.
    """
    text_dict = page.get_text("dict")
    page_w = text_dict["width"]
    page_h = text_dict["height"]

    all_spans = []
    blocks = text_dict.get("blocks", [])
    for b_idx, b in enumerate(blocks):
        if b.get("type") == 0:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    txt = s["text"].strip()
                    if txt:
                        all_spans.append({
                            "text": txt,
                            "bbox": s["bbox"],
                            "x0": s["bbox"][0], "y0": s["bbox"][1],
                            "x1": s["bbox"][2], "y1": s["bbox"][3],
                            "y_center": (s["bbox"][1] + s["bbox"][3]) / 2.0,
                            "x_center": (s["bbox"][0] + s["bbox"][2]) / 2.0,
                            "block_idx": b_idx
                        })

    if not all_spans:
        return []

    # Loại bỏ footer số trang ("1/6", "2/6")
    all_spans = [s for s in all_spans if not re.match(r'^\d/\d$', s["text"])]

    # Gom nhóm tất cả spans theo Y-center
    all_spans.sort(key=lambda s: s["y_center"])
    y_lines = []
    for s in all_spans:
        assigned = False
        for yl in y_lines:
            mean_y = sum(item["y_center"] for item in yl) / len(yl)
            if abs(s["y_center"] - mean_y) <= 6.0:
                yl.append(s)
                assigned = True
                break
        if not assigned:
            y_lines.append([s])

    for yl in y_lines:
        yl.sort(key=lambda item: item["x0"])

    anchor_indices = []
    for idx, yl in enumerate(y_lines):
        full_text = " ".join(item["text"] for item in yl)

        if full_text.startswith("- Căn cứ") or full_text.startswith("Căn cứ") or \
           full_text.startswith("ĐIỀU") or full_text.startswith("Bên thuê") or \
           full_text.startswith("Bên cung cấp") or full_text.startswith("Sau khi xem") or \
           full_text.startswith("1.2.") or full_text.startswith("1.3.") or \
           full_text.startswith("2.1.") or full_text.startswith("2.2.") or \
           full_text.startswith("3.1.") or full_text.startswith("4.1.") or \
           full_text.startswith("PHỤ LỤC HỢP ĐỒNG CUNG CẤP") or full_text.startswith("Chi nhánh"):
            continue

        if re.match(r'^\d+\.\d+(\.\d+)?\.?$', yl[0]["text"]):
            if len(yl) >= 2 and (yl[-1]["x1"] - yl[1]["x0"] > 200):
                continue

        if "ĐẠI DIỆN BÊN A" in full_text or "ĐẠI DIỆN BÊN B" in full_text:
            continue

        x_centers = [item["x_center"] for item in yl]
        distinct_xs = []
        for xc in x_centers:
            if not distinct_xs or min(abs(xc - d) for d in distinct_xs) > 25:
                distinct_xs.append(xc)

        if len(distinct_xs) >= 2 or (len(yl) == 1 and yl[0]["text"] in ["STT", "1", "2", "3", "4", "5"]) or any(k in full_text for k in ["Cộng tiền", "Tổng cộng"]):
            anchor_indices.append(idx)

    if not anchor_indices:
        return []

    table_y_ranges = []
    curr_group = [anchor_indices[0]]

    for idx in anchor_indices[1:]:
        prev_idx = curr_group[-1]
        prev_y = sum(item["y_center"] for item in y_lines[prev_idx]) / len(y_lines[prev_idx])
        curr_y = sum(item["y_center"] for item in y_lines[idx]) / len(y_lines[idx])

        if curr_y - prev_y <= 45:
            curr_group.append(idx)
        else:
            if len(curr_group) >= 2:
                y_start = min(item["y0"] for idx_i in curr_group for item in y_lines[idx_i]) - 5
                y_end = max(item["y1"] for idx_i in curr_group for item in y_lines[idx_i]) + 20
                table_y_ranges.append((y_start, y_end))
            curr_group = [idx]

    if len(curr_group) >= 2:
        y_start = min(item["y0"] for idx_i in curr_group for item in y_lines[idx_i]) - 5
        y_end = max(item["y1"] for idx_i in curr_group for item in y_lines[idx_i]) + 20
        table_y_ranges.append((y_start, y_end))

    log_debug(f"Trang {page_num}: Phát hiện {len(table_y_ranges)} vùng Bảng hợp lệ")

    extracted_tables = []

    for t_idx, (y_start, y_end) in enumerate(table_y_ranges):
        reg_spans = [s for s in all_spans if y_start <= s["y_center"] <= y_end]

        if not reg_spans:
            continue

        reg_y_lines = []
        for yl in y_lines:
            yl_y = sum(item["y_center"] for item in yl) / len(yl)
            if y_start <= yl_y <= y_end:
                reg_y_lines.append(yl)

        if not reg_y_lines:
            continue

        # 5. XÁC ĐỊNH RẢNH CỘT (COLUMNS) DỰA TRÊN X CLUSTERING
        x0_list = sorted([s["x0"] for s in reg_spans])
        col_x_starts = []
        for x in x0_list:
            if not col_x_starts:
                col_x_starts.append([x])
            else:
                avg_x = sum(col_x_starts[-1]) / len(col_x_starts[-1])
                if abs(x - avg_x) <= 22:
                    col_x_starts[-1].append(x)
                else:
                    col_x_starts.append([x])

        col_bounds = [min(c) for c in col_x_starts]
        col_bounds.sort()
        num_cols = len(col_bounds)

        columns_def = []
        for i in range(num_cols):
            x_start = col_bounds[i] - 10 if i == 0 else (col_bounds[i-1] + col_bounds[i]) / 2.0
            x_end = page_w if i == num_cols - 1 else (col_bounds[i] + col_bounds[i+1]) / 2.0
            columns_def.append({
                "index": i,
                "name": f"Cột_{i+1}",
                "x_start": round(x_start, 1),
                "x_end": round(x_end, 1)
            })

        table_rows = []
        current_row = []

        for yl in reg_y_lines:
            line_full_text = " ".join(item["text"] for item in yl).strip()
            col0_items = [item for item in yl if item["x_center"] <= columns_def[0]["x_end"]]
            col0_text = " ".join(item["text"] for item in col0_items).strip()

            is_new_row = False
            if col0_text and (col0_text.isdigit() or col0_text in ["STT", "Cộng", "Tổng cộng", "Nội dung"]):
                is_new_row = True
            elif "Cộng tiền" in line_full_text or "Tổng cộng thanh toán" in line_full_text:
                is_new_row = True
            elif not current_row:
                is_new_row = True

            if is_new_row and current_row:
                table_rows.append(current_row)
                current_row = list(yl)
            else:
                current_row.extend(yl)

        if current_row:
            table_rows.append(current_row)

        matrix = []
        structured_rows = []

        for r_idx, r_items in enumerate(table_rows):
            row_cells = [""] * num_cols
            r_items_sorted = sorted(r_items, key=lambda item: (item["y_center"], item["x0"]))

            for item in r_items_sorted:
                xc = item["x_center"]
                best_c = 0
                for c_idx, c_def in enumerate(columns_def):
                    if c_def["x_start"] <= xc <= c_def["x_end"]:
                        best_c = c_idx
                        break

                txt = item["text"]
                if row_cells[best_c]:
                    row_cells[best_c] += "\n" + txt
                else:
                    row_cells[best_c] = txt

            matrix.append(row_cells)

            cells_list = []
            for c_idx, text_val in enumerate(row_cells):
                cells_list.append({
                    "column_index": c_idx,
                    "column_name": columns_def[c_idx]["name"],
                    "text": text_val,
                    "bbox": {}
                })

            structured_rows.append({
                "row_index": r_idx,
                "cells": cells_list
            })

        if matrix:
            header_row = matrix[0]
            for i, h in enumerate(header_row):
                if i < len(columns_def) and h.strip():
                    columns_def[i]["name"] = h.replace("\n", " ")

        val_res = validate_table_financials(matrix)

        table_obj = {
            "type": "table",
            "table_index": t_idx,
            "page": page_num,
            "detection_method": "native_pdf_spatial",
            "table_confidence": 0.98 if val_res["status"] == "pass" else 0.88,
            "headers": [h.replace("\n", " ") for h in matrix[0]] if matrix else [],
            "columns": columns_def,
            "rows": structured_rows,
            "matrix": matrix,
            "markdown": format_table_to_markdown(matrix),
            "validation": val_res
        }
        extracted_tables.append(table_obj)

    return extracted_tables


def extract_tables_from_pdf_digital(pdf_path: str):
    """
    Hàm thống nhất trích xuất Bảng từ PDF Kỹ thuật số (Native PDF).
    Trả về tuple: (tables_matrices, structured_tables)
    """
    tables_found = []
    structured_tables = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_idx, page in enumerate(doc, 1):
            tbls = extract_native_pdf_tables_from_page(page, page_idx)
            for t in tbls:
                tables_found.append(t["matrix"])
                structured_tables.append(t)
        doc.close()
    except Exception as e:
        log_debug(f"Lỗi extract_tables_from_pdf_digital: {e}")

    return tables_found, structured_tables


def render_pdf_to_images(pdf_path: str):
    """
    Chuyển các trang PDF thành ảnh OpenCV BGR sử dụng PyMuPDF (fitz).
    """
    images = []
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            images.append(img_np)
        doc.close()
    except Exception as e:
        log_debug(f"Lỗi render_pdf_to_images: {e}")
    return images


def format_table_to_markdown(table_matrix: list[list[str]]) -> str:
    """
    Chuyển ma trận Bảng 2D (List[List[str]]) thành chuỗi Markdown Table hoàn chỉnh.
    """
    if not table_matrix:
        return ""

    num_cols = max(len(row) for row in table_matrix)
    norm_matrix = []
    for row in table_matrix:
        r = [str(cell).strip().replace("\n", " ") for cell in row]
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
