import os
import shutil
import sys
import pypdf
import unicodedata
import re

# ---------------------------------------------------------------------------
# Dùng chung EasyOCR singleton với document_reader.py — tránh load model 2 lần
# ---------------------------------------------------------------------------
def get_ocr_reader():
    """
    Proxy đến singleton trong document_reader để đảm bảo chỉ có 1 Reader toàn ứng dụng.
    """
    from document_reader import get_ocr_reader as _get_reader
    return _get_reader()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Trích xuất text từ file PDF sử dụng pypdf.
    """
    try:
        reader = pypdf.PdfReader(file_path)
        text_pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_pages.append(t)
        return "\n".join(text_pages)
    except Exception as e:
        print(f"Lỗi khi đọc file PDF {file_path}: {e}")
        return ""


def extract_text_from_image(file_path: str) -> str:
    """
    Trích xuất text từ file ảnh (.png, .jpg, .jpeg, .webp, .bmp) sử dụng EasyOCR.
    Dùng chung reader singleton — không khởi tạo lại.
    """
    try:
        reader = get_ocr_reader()
        results = reader.readtext(file_path, detail=0)
        return " ".join(results)
    except Exception as e:
        print(f"Lỗi khi OCR file ảnh {file_path}: {e}")
        return ""


def _fast_extract_text_for_classify(file_path: str, max_chars: int = 800) -> str:
    """
    Fast-path: trích xuất nhanh ~800 ký tự đầu của tài liệu chỉ phục vụ phân loại.
    - PDF text: chỉ đọc trang đầu bằng pypdf (rất nhanh, không cần EasyOCR)
    - PDF scan: thử pypdf trước, nếu rỗng mới OCR trang đầu
    - Ảnh: OCR nhanh bằng EasyOCR (readtext toàn bộ, nhanh hơn read_document đầy đủ)
    - DOCX/XLSX: đọc native bình thường
    Không gọi read_document() đầy đủ để tránh tốn công.
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == '.pdf':
            # Thử đọc native text của trang đầu (cực nhanh)
            try:
                import pymupdf
                doc = pymupdf.open(file_path)
                for page_idx in range(min(2, len(doc))):
                    t = doc[page_idx].get_text("text") or ""
                    text += t
                    if len(text) >= max_chars:
                        break
                doc.close()
            except Exception:
                pass

            # Nếu không có text native → fallback pypdf
            if len(text.strip()) < 20:
                text = extract_text_from_pdf(file_path)

        elif ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
            # OCR nhanh: readtext với detail=0, không preprocess nặng
            try:
                reader = get_ocr_reader()
                results = reader.readtext(file_path, detail=0)
                text = " ".join(results)
            except Exception as e:
                print(f"  [Classify] OCR error: {e}")

        elif ext == '.docx':
            try:
                import docx
                doc = docx.Document(file_path)
                parts = []
                for para in doc.paragraphs[:30]:
                    if para.text.strip():
                        parts.append(para.text)
                text = "\n".join(parts)
            except Exception:
                pass

        elif ext == '.xlsx':
            try:
                import openpyxl
                wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
                parts = []
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(max_row=10, values_only=True):
                        parts.append(" ".join(str(c) for c in row if c is not None))
                    break  # chỉ sheet đầu
                wb.close()
                text = "\n".join(parts)
            except Exception:
                pass

    except Exception as e:
        print(f"  [Classify] fast_extract error for '{os.path.basename(file_path)}': {e}")

    return text[:max_chars]


def extract_document_text(file_path: str) -> str:
    """
    Trích xuất văn bản từ tài liệu (hỗ trợ PDF, DOCX, XLSX và các định dạng ảnh).
    Sử dụng fast-path để phân loại, không gọi read_document() đầy đủ.
    """
    return _fast_extract_text_for_classify(file_path)


def remove_accents(text: str) -> str:
    """
    Loại bỏ dấu tiếng Việt để so sánh chuỗi không dấu (tránh lỗi font/OCR mất dấu).
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = unicodedata.normalize('NFC', text)
    return text.replace('Đ', 'D').replace('đ', 'd')


def classify_document(file_path: str) -> tuple[str, float]:
    """
    Phân loại tài liệu (PDF, Word, Excel, Ảnh) thông minh dựa trên:
    - Loại file & Tên file
    - Từ khóa Tiếng Việt có dấu & không dấu (loại bỏ lỗi mã hóa font/OCR)
    - Biểu thức chính quy (Regex) & Hệ thống chấm điểm trọng số (Scoring System).
    Dùng fast-path extract chỉ lấy 800 ký tự đầu — không cần OCR toàn bộ.
    """
    filename = os.path.basename(file_path).lower()
    full_text = extract_document_text(file_path)

    lines = [l.strip() for l in full_text.split('\n') if l.strip()]
    header_lines = lines[:20] if lines else []
    header_text = "\n".join(header_lines).upper() if header_lines else full_text.upper()
    header_unaccented = remove_accents(header_text).upper()

    full_text_upper = full_text.upper()
    full_text_unaccented = remove_accents(full_text).upper()
    filename_unaccented = remove_accents(filename).lower()

    # Điểm số cho từng danh mục
    scores = {
        'hop_dong': 0,
        'hoa_don': 0,
        'chung_tu': 0,
        'anh_chuyen_khoan': 0
    }

    # 1. KIỂM TRA ẢNH CHUYỂN KHOẢN / BIÊN NHẬN NGÂN HÀNG
    transfer_keywords = [
        "CHUYEN KHOAN THANH CONG", "GIAO DICH THANH CONG", "CHUYEN TIEN THANH CONG",
        "BIEN NHAN CHUYEN TIEN", "XAC NHAN CHUYEN TIEN", "THONG TIN CHUYEN KHOAN",
        "NOI DUNG CHUYEN KHOAN", "LENH CHUYEN TIEN", "UY NHIEM CHI", "SO THAM CHIEU",
        "MA GIAO DICH", "NGAN HANG THU HUONG", "TAI KHOAN THU HUONG", "VIETQR", "MOMO", "ZALOPAY"
    ]
    for kw in transfer_keywords:
        if kw in full_text_unaccented:
            scores['anh_chuyen_khoan'] += 45
        if kw in header_unaccented:
            scores['anh_chuyen_khoan'] += 20

    fn_transfer_clues = [
        "chuyen_khoan", "chuyenkhoan", "chuyen_tien", "giao_dich",
        "bien_nhan", "ck_", "receipt", "vietcombank", "vcb", "digibank"
    ]
    if any(c in filename_unaccented for c in fn_transfer_clues):
        scores['anh_chuyen_khoan'] += 35

    # Một số ảnh chụp màn hình ngân hàng có thể không OCR được (nền tối,
    # chữ sáng hoặc lớp thông báo che phần đầu ảnh). Tên file thường vẫn
    # giữ lại dấu hiệu VCB/Digibank, nên dùng thêm tín hiệu nhẹ này thay vì
    # trả thẳng về "khac" khi OCR không có kết quả.
    if any(c in filename_unaccented for c in ("vietcombank", "vcb", "digibank")):
        scores['anh_chuyen_khoan'] += 20

    # Các nhãn thực tế thường xuất hiện trong biên nhận chuyển khoản.
    transfer_label_pairs = [
        ("TAI KHOAN NHAN", "NGAN HANG NHAN"),
        ("SO TAI KHOAN NHAN", "TEN NGUOI NHAN"),
        ("GIAO DICH THANH CONG", "VND"),
    ]
    for left, right in transfer_label_pairs:
        if left in full_text_unaccented and right in full_text_unaccented:
            scores['anh_chuyen_khoan'] += 25

    if ("SO TIEN" in full_text_unaccented or "TAI KHOAN" in full_text_unaccented) and \
       ("THANH CONG" in full_text_unaccented or "NGAN HANG" in full_text_unaccented or "VIETCOMBANK" in full_text_unaccented):
        scores['anh_chuyen_khoan'] += 40

    # 2. KIỂM TRA HỢP ĐỒNG (CONTRACT)
    contract_header_keywords = [
        "HOP DONG", "CONG HOA XA HOI CHU NGHIA VIET NAM", "BIEN BAN GIAO NHAN",
        "CONTRACT", "AGREEMENT", "BEN A", "BEN B"
    ]
    for kw in contract_header_keywords:
        if kw in header_unaccented:
            scores['hop_dong'] += 40
        elif kw in full_text_unaccented:
            scores['hop_dong'] += 20

    fn_contract_clues = ["hop_dong", "hopdong", "hdmb", "hđ", "contract", "agreement"]
    if any(c in filename_unaccented for c in fn_contract_clues):
        scores['hop_dong'] += 35

    if re.search(r'DIEU\s+\d+', full_text_unaccented) or re.search(r'ĐIỀU\s+\d+', full_text_upper):
        scores['hop_dong'] += 25

    # 3. KIỂM TRA HÓA ĐƠN (INVOICE)
    invoice_keywords = [
        "HOA DON", "HOÁ ĐƠN", "VAT INVOICE", "INVOICE", "MA SO THUE",
        "GTGT", "GIA TRI GIA TANG", "TEN DON VI BAN", "KY HIEU"
    ]
    for kw in invoice_keywords:
        if kw in header_unaccented:
            scores['hoa_don'] += 45
        elif kw in full_text_unaccented:
            scores['hoa_don'] += 20

    fn_invoice_clues = ["hoadon", "hoa_don", "invoice", "vat", "bill"]
    if any(c in filename_unaccented for c in fn_invoice_clues):
        scores['hoa_don'] += 35

    # 4. KIỂM TRA CHỨNG TỪ / ĐƠN HÀNG / BÁO CÁO (VOUCHER / ORDER / REPORT)
    voucher_keywords = [
        "DON DAT HANG", "DON HANG", "CHUNG TU", "PHIEU XUAT KHO", "PHIEU NHAP KHO",
        "PHIEU THU", "PHIEU CHI", "SALES ORDER", "PURCHASE ORDER", "BAO CAO",
        "MA HD", "DOANH THU", "SO TIEN", "KHACH HANG", "STT"
    ]
    for kw in voucher_keywords:
        if kw in header_unaccented:
            scores['chung_tu'] += 35
        elif kw in full_text_unaccented:
            scores['chung_tu'] += 15

    fn_voucher_clues = ["so-", "chung_tu", "chungtu", "don_hang", "order", "excel", "report", "pxk", "pnk"]
    if any(c in filename_unaccented for c in fn_voucher_clues):
        scores['chung_tu'] += 35

    # Đuôi file đặc thù (Excel thường là chứng từ / báo cáo doanh thu / bảng kê)
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        scores['chung_tu'] += 20

    # Tìm danh mục có điểm cao nhất
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # Nếu không có điểm nào hoặc điểm quá thấp (< 15)
    if best_score < 15:
        # Fallback 1: Thử xem có chữ "HOP" + "DONG" / "HOA" + "DON" / "CHUNG" + "TU"
        if "HOP" in full_text_unaccented and "DONG" in full_text_unaccented:
            return ("hop_dong", 70.0)
        if "HOA" in full_text_unaccented and "DON" in full_text_unaccented:
            return ("hoa_don", 70.0)
        if "CHUNG" in full_text_unaccented and "TU" in full_text_unaccented:
            return ("chung_tu", 70.0)
        return ("khac", 0.0)

    # Tính phần trăm độ tin cậy dựa trên điểm số (tối đa 99.0%)
    confidence = min(99.0, round(50.0 + (best_score * 0.8), 1))
    return (best_cat, confidence)


def organize_documents(src_dir: str, target_dir: str = None):
    """
    Phân loại các tài liệu từ src_dir và di chuyển vào các folder tương ứng trong target_dir.
    Nếu target_dir không được truyền vào, sẽ tạo các folder con bên trong src_dir.
    """
    if target_dir is None:
        target_dir = src_dir

    if not os.path.exists(src_dir):
        print(f"Thư mục nguồn '{src_dir}' không tồn tại!")
        return []

    categories = {
        'hop_dong': 'Hợp đồng',
        'hoa_don': 'Hóa đơn',
        'chung_tu': 'Chứng từ',
        'anh_chuyen_khoan': 'Ảnh chuyển khoản'
    }

    folder_paths = {}
    for cat in categories:
        target_folder = os.path.join(target_dir, cat)
        os.makedirs(target_folder, exist_ok=True)
        folder_paths[cat] = target_folder

    moved_files = []
    supported_exts = ('.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg', '.webp', '.bmp')

    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(supported_exts)]

    if not files:
        print(f"Không tìm thấy tài liệu phù hợp (PDF/Ảnh) trực tiếp trong thư mục '{src_dir}'.")
        return []

    print(f"Phát hiện {len(files)} tài liệu trong '{src_dir}' cần phân loại...")

    for filename in files:
        file_path = os.path.join(src_dir, filename)
        category, confidence = classify_document(file_path)

        if category in folder_paths:
            dest_folder = folder_paths[category]
            dest_path = os.path.join(dest_folder, filename)
            shutil.move(file_path, dest_path)
            moved_files.append({
                'file_name': filename,
                'category': category,
                'category_name': categories[category],
                'confidence': confidence,
                'dest_folder': dest_folder,
                'dest_path': dest_path
            })
            print(f"[✓] Đã chuyển '{filename}' -> [{categories[category]}] ({dest_folder}) (Độ tin cậy: {confidence}%)")
        else:
            print(f"[?] Không thể phân loại file '{filename}' (Độ tin cậy: {confidence}%)")

    return moved_files


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    doc_directory = r"d:\AI\ocr\doc"
    print("=" * 60)
    print(f"BẮT ĐẦU PHÂN LOẠI VÀ CHUYỂN CÁC TÀI LIỆU TRONG THƯ MỤC: {doc_directory}")
    print("=" * 60)
    results = organize_documents(doc_directory)
    print("\n=> HOÀN THÀNH PHÂN LOẠI VÀ DI CHUYỂN FILE!")
