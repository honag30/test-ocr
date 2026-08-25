import os
import re
import cv2
import numpy as np
import pymupdf  # fitz
import pdfplumber

from normalizer import normalize_vietnamese_text, is_usable_text, detect_headers_footers
from image_processor import preprocess_image
from ocr_utils import group_results_by_line
from table_detector import extract_table_from_raw_results, format_table_to_markdown, extract_native_pdf_tables_from_page

# Lazy-loaded EasyOCR reader
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[System] Initializing EasyOCR Reader (vi, en)...")
        _ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)
    return _ocr_reader

def classify_text_element(line: str) -> dict:
    """
    Phân loại từng dòng/đoạn văn bản thành cấu trúc:
    heading | list | paragraph
    """
    line_clean = line.strip()
    if not line_clean:
        return {"type": "paragraph", "text": ""}

    # 1. Heading (Ví dụ: ĐIỀU 1, MỤC I, CHƯƠNG II, hoặc IN HOA HOÀN TOÀN có chữ ngắn)
    heading_patterns = [
        r'^(ĐIỀU|Đieu|CHƯƠNG|Chương|MỤC|Mục|PHẦN|Phần)\s+\d+',
        r'^[I|V|X]+\.\s+',
        r'^\d+\.\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]',
    ]
    is_heading = any(re.search(pat, line_clean) for pat in heading_patterns)
    if not is_heading and line_clean.isupper() and len(line_clean) < 80 and not line_clean.endswith('.'):
        is_heading = True

    if is_heading:
        return {"type": "heading", "text": line_clean}

    # 2. List Item (Ví dụ: - ..., + ..., 1.1 ..., a) ...)
    list_patterns = [
        r'^[•\-\*\+]\s+',
        r'^\d+\.\d+(\.\d+)?\s+',
        r'^[a-z]\)\s+',
    ]
    if any(re.search(pat, line_clean) for pat in list_patterns):
        return {"type": "list", "text": line_clean}

    return {"type": "paragraph", "text": line_clean}

def read_pdf(file_path: str, include_header: True, include_footer: True) -> dict:
    """
    Đọc file PDF theo từng trang:
    - Trang nào có text layer hợp lệ -> Native Text & Table Extraction
    - Trang nào không có text / scan / font lỗi -> Render thành ảnh -> EasyOCR
    - Nhận diện Header/Footer và hỗ trợ bao gồm hoặc loại bỏ.
    """
    doc = pymupdf.open(file_path)
    total_pages = len(doc)
    print(f"[PDF] Detected {total_pages} page(s) in '{os.path.basename(file_path)}'")

    page_results = []
    page_texts_raw = []

    native_pages_count = 0
    ocr_pages_count = 0
    all_tables = []

    for idx in range(total_pages):
        page_num = idx + 1
        page = doc[idx]
        native_text = page.get_text("text") or ""
        native_text = normalize_vietnamese_text(native_text)

        tables_in_page = []

        # 1. Kiểm tra chất lượng Native Text
        usable = is_usable_text(native_text, min_length=20)

        if usable:
            native_pages_count += 1
            print(f"  [Page {page_num}/{total_pages}] Native text extraction (Success)")

            # Extract bảng bằng thuật toán Native PDF Spatial Classification
            tables_in_page = extract_native_pdf_tables_from_page(page, page_num)

            lines = [l.strip() for l in native_text.split('\n') if l.strip()]
            elements = [classify_text_element(l) for l in lines]

            page_results.append({
                "page": page_num,
                "method": "native_text",
                "confidence": None,
                "text": native_text,
                "lines": lines,
                "elements": elements,
                "tables": tables_in_page
            })
            page_texts_raw.append(native_text)
        else:
            ocr_pages_count += 1
            print(f"  [Page {page_num}/{total_pages}] No usable text layer -> Fallback to OCR")

            # Render trang này thành ảnh để OCR
            pix = page.get_pixmap(dpi=200)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            processed_img = preprocess_image(img_np, image_input_is_np=True)

            reader = get_ocr_reader()
            raw_results = reader.readtext(
                processed_img,
                decoder='beamsearch',
                beamWidth=5,
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.4,
                mag_ratio=1.5
            )

            ocr_lines, ocr_confs = group_results_by_line(raw_results, y_tolerance=20)
            avg_conf = round(sum(ocr_confs) / max(1, len(ocr_confs)) * 100.0, 2) if ocr_confs else 0.0

            # Nếu chưa có bảng từ pdfplumber, thử phát hiện bảng từ OCR + OpenCV grid
            if not tables_in_page:
                matrix, found, struct_t = extract_table_from_raw_results(raw_results, image=processed_img)
                if found:
                    headers = matrix[0] if matrix else []
                    rows = matrix[1:] if len(matrix) > 1 else []
                    table_obj = {
                        "type": "table",
                        "page": page_num,
                        "detection_method": struct_t.get("detection_method", "ocr_table"),
                        "headers": headers,
                        "rows": rows,
                        "matrix": matrix,
                        "markdown": format_table_to_markdown(matrix)
                    }
                    tables_in_page.append(table_obj)

            # Giải phóng ảnh khỏi bộ nhớ
            del pix, img_np, processed_img

            page_text_ocr = normalize_vietnamese_text("\n".join(ocr_lines))
            elements = [classify_text_element(l) for l in ocr_lines]

            page_results.append({
                "page": page_num,
                "method": "ocr",
                "confidence": avg_conf,
                "text": page_text_ocr,
                "lines": ocr_lines,
                "confidences": ocr_confs,
                "elements": elements,
                "tables": tables_in_page
            })
            page_texts_raw.append(page_text_ocr)

        if tables_in_page:
            all_tables.extend(tables_in_page)

    doc.close()

    # Determine source_type & extraction_method
    if ocr_pages_count == 0:
        source_type = "pdf_text"
        extraction_method = "native_text"
    elif native_pages_count == 0:
        source_type = "pdf_scan"
        extraction_method = "ocr"
    else:
        source_type = "pdf_mixed"
        extraction_method = "mixed"

    # Header / Footer detection
    headers_set, footers_set = detect_headers_footers(page_texts_raw)

    full_text_lines = []
    for pr in page_results:
        for l in pr.get("lines", []):
            l_clean = l.strip()
            if not include_header and l_clean in headers_set:
                continue
            if not include_footer and l_clean in footers_set:
                continue
            full_text_lines.append(l_clean)

    full_text = normalize_vietnamese_text("\n".join(full_text_lines))

    return {
        "document_type": "pdf",
        "source_type": source_type,
        "extraction_method": extraction_method,
        "has_table": bool(all_tables),
        "total_pages": total_pages,
        "headers_detected": list(headers_set),
        "footers_detected": list(footers_set),
        "pages": page_results,
        "full_text": full_text,
        "tables": all_tables
    }

def read_docx(file_path: str) -> dict:
    """
    Đọc file DOCX trực tiếp:
    - Trích xuất Paragraphs, Headings, Lists, Tables
    - Không chuyển sang ảnh để OCR.
    """
    import docx
    print(f"[DOCX] Reading natively: '{os.path.basename(file_path)}'")
    doc = docx.Document(file_path)

    elements = []
    full_text_lines = []
    tables_list = []

    # Duyệt qua các block theo đúng thứ tự xuất hiện trong docx
    for element in doc.element.body:
        if element.tag.endswith('p'): # Paragraph
            p = docx.text.paragraph.Paragraph(element, doc)
            text = normalize_vietnamese_text(p.text)
            if not text:
                continue

            style_name = p.style.name.lower() if p.style else ""
            if "heading" in style_name or "title" in style_name:
                elem_type = "heading"
            elif "list" in style_name:
                elem_type = "list"
            else:
                elem_type = classify_text_element(text)["type"]

            elements.append({"type": elem_type, "text": text})
            full_text_lines.append(text)

        elif element.tag.endswith('tbl'): # Table
            tbl = docx.table.Table(element, doc)
            matrix = []
            for row in tbl.rows:
                row_cells = [normalize_vietnamese_text(cell.text) for cell in row.cells]
                if any(row_cells):
                    matrix.append(row_cells)

            if len(matrix) >= 1:
                headers = matrix[0]
                rows = matrix[1:] if len(matrix) > 1 else []
                table_obj = {
                    "type": "table",
                    "page": 1,
                    "detection_method": "docx_native",
                    "headers": headers,
                    "rows": rows,
                    "matrix": matrix,
                    "markdown": format_table_to_markdown(matrix)
                }
                tables_list.append(table_obj)

                # Cũng thêm text của bảng vào full_text
                table_text = "\n".join(["\t".join(r) for r in matrix])
                elements.append({"type": "table", "text": table_text, "headers": headers, "rows": rows})
                full_text_lines.append(table_text)

    full_text = normalize_vietnamese_text("\n".join(full_text_lines))

    return {
        "document_type": "docx",
        "source_type": "docx",
        "extraction_method": "native_text",
        "has_table": bool(tables_list),
        "total_pages": 1,
        "pages": [{
            "page": 1,
            "method": "native_text",
            "confidence": None,
            "text": full_text,
            "lines": full_text_lines,
            "elements": elements,
            "tables": tables_list
        }],
        "full_text": full_text,
        "tables": tables_list
    }

def read_xlsx(file_path: str) -> dict:
    """
    Đọc file XLSX trực tiếp:
    - Trích xuất các Sheet, Row, Column, Cell
    - Không chuyển sang ảnh để OCR.
    """
    import openpyxl
    print(f"[XLSX] Reading natively: '{os.path.basename(file_path)}'")
    wb = openpyxl.load_workbook(file_path, data_only=True)

    tables_list = []
    full_text_lines = []
    elements = []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        matrix = []
        for row in sheet.iter_rows(values_only=True):
            row_str = [normalize_vietnamese_text(str(cell)) if cell is not None else "" for cell in row]
            if any(row_str):
                matrix.append(row_str)

        if matrix:
            headers = matrix[0]
            rows = matrix[1:] if len(matrix) > 1 else []
            table_obj = {
                "type": "table",
                "sheet_name": sheet_name,
                "detection_method": "xlsx_native",
                "headers": headers,
                "rows": rows,
                "matrix": matrix,
                "markdown": format_table_to_markdown(matrix)
            }
            tables_list.append(table_obj)

            sheet_text = f"=== Sheet: {sheet_name} ===\n" + "\n".join(["\t".join(r) for r in matrix])
            elements.append({"type": "heading", "text": f"Sheet: {sheet_name}"})
            elements.append({"type": "table", "text": sheet_text, "headers": headers, "rows": rows})
            full_text_lines.append(sheet_text)

    wb.close()
    full_text = normalize_vietnamese_text("\n".join(full_text_lines))

    return {
        "document_type": "xlsx",
        "source_type": "xlsx",
        "extraction_method": "table_extraction",
        "has_table": bool(tables_list),
        "total_pages": len(tables_list),
        "pages": [{
            "page": idx + 1,
            "method": "table_extraction",
            "confidence": None,
            "sheet_name": tbl.get("sheet_name"),
            "text": tbl.get("markdown"),
            "tables": [tbl]
        } for idx, tbl in enumerate(tables_list)],
        "full_text": full_text,
        "tables": tables_list
    }

def read_image(file_path: str) -> dict:
    """
    Đọc file ảnh (PNG, JPG, JPEG, WEBP, BMP):
    - Tiền xử lý ảnh (Deskew, Shadow/BG Removal, Denoise)
    - OCR với EasyOCR
    - Phát hiện Bảng bằng OpenCV Morphology Grid & Spatial Clustering
    """
    print(f"[Image] Processing & OCR: '{os.path.basename(file_path)}'")
    processed_img = preprocess_image(file_path)

    reader = get_ocr_reader()
    raw_results = reader.readtext(
        processed_img,
        decoder='beamsearch',
        beamWidth=5,
        text_threshold=0.4,
        low_text=0.2,
        link_threshold=0.4,
        mag_ratio=1.5
    )

    lines, confidences = group_results_by_line(raw_results, y_tolerance=20)
    avg_conf = round(sum(confidences) / max(1, len(confidences)) * 100.0, 2) if confidences else 0.0

    table_matrix, found, struct_t = extract_table_from_raw_results(raw_results, image=processed_img)
    tables_list = []
    if found:
        headers = table_matrix[0] if table_matrix else []
        rows = table_matrix[1:] if len(table_matrix) > 1 else []
        tables_list.append({
            "type": "table",
            "page": 1,
            "detection_method": struct_t.get("detection_method", "ocr_grid"),
            "headers": headers,
            "rows": rows,
            "matrix": table_matrix,
            "markdown": format_table_to_markdown(table_matrix)
        })
        print("  -> [✓] Table detected in image!")

    full_text = normalize_vietnamese_text("\n".join(lines))
    elements = [classify_text_element(l) for l in lines]

    return {
        "document_type": "image",
        "source_type": "image",
        "extraction_method": "ocr",
        "has_table": bool(tables_list),
        "total_pages": 1,
        "pages": [{
            "page": 1,
            "method": "ocr",
            "confidence": avg_conf,
            "text": full_text,
            "lines": lines,
            "confidences": confidences,
            "elements": elements,
            "tables": tables_list
        }],
        "full_text": full_text,
        "tables": tables_list
    }

def read_document(file_path: str, include_header: bool = True, include_footer: bool = True) -> dict:
    """
    Hàm entry point thống nhất đọc mọi loại tài liệu điện tử.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        res = read_pdf(file_path, include_header, include_footer)
    elif ext == '.docx':
        res = read_docx(file_path)
    elif ext == '.xlsx':
        res = read_xlsx(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
        res = read_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    res["original_filename"] = os.path.basename(file_path)
    res["file_path"] = file_path
    print(f"[Document] Extraction completed for '{os.path.basename(file_path)}' (Source: {res['source_type']}, Method: {res['extraction_method']})")
    return res
