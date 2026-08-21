import os
import shutil
import sys
import pypdf

# Lazy-loaded EasyOCR reader
_ocr_reader = None

def get_ocr_reader():
    """
    Khởi tạo EasyOCR Reader một lần duy nhất khi cần đọc text từ ảnh.
    """
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("-> Đang khởi tạo EasyOCR Reader...")
        _ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)
    return _ocr_reader

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
    """
    try:
        reader = get_ocr_reader()
        results = reader.readtext(file_path, detail=0)
        return " ".join(results)
    except Exception as e:
        print(f"Lỗi khi OCR file ảnh {file_path}: {e}")
        return ""

def extract_document_text(file_path: str) -> str:
    """
    Trích xuất văn bản từ tài liệu (hỗ trợ cả PDF và các định dạng ảnh).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
        return extract_text_from_image(file_path)
    return ""

def classify_document(file_path: str) -> tuple[str, float]:
    """
    Phân loại tài liệu (PDF hoặc Ảnh) và trả về tuple: (mã_danh_mục, độ_tin_cậy_%).
    Các nhóm:
    - 'hop_dong'        : Hợp đồng
    - 'hoa_don'         : Hóa đơn
    - 'chung_tu'        : Chứng từ / Đơn hàng
    - 'anh_chuyen_khoan': Ảnh / Biên nhận chuyển khoản
    - 'khac'            : Chưa xác định
    """
    filename = os.path.basename(file_path).lower()
    full_text = extract_document_text(file_path)
    
    lines = [l.strip() for l in full_text.split('\n') if l.strip()]
    header_text = "\n".join(lines[:20]).upper() if lines else full_text.upper()
    full_text_upper = full_text.upper()
    
    # 1. Kiểm tra Ảnh / Biên nhận chuyển khoản
    transfer_keywords = [
        "CHUYỂN KHOẢN THÀNH CÔNG", "CHUYEN KHOAN THANH CONG",
        "GIAO DỊCH THÀNH CÔNG", "GIAO DICH THANH CONG",
        "CHUYỂN TIỀN THÀNH CÔNG", "CHUYEN TIEN THANH CONG",
        "BIÊN NHẬN CHUYỂN TIỀN", "XÁC NHẬN CHUYỂN TIỀN",
        "THÔNG TIN CHUYỂN KHOẢN", "NỘI DUNG CHUYỂN KHOẢN",
        "LỆNH CHUYỂN TIỀN", "ỦY NHIỆM CHI", "SỐ THAM CHIẾU",
        "MÃ GIAO DỊCH", "NGÂN HÀNG THỤ HƯỞNG", "TÀI KHOẢN THỤ HƯỞNG",
        "VIETQR", "MOMO", "ZALOPAY"
    ]
    fn_transfer_clues = ["chuyen_khoan", "chuyenkhoan", "chuyen_tien", "giao_dich", "bien_nhan", "ck_"]
    
    if any(k in full_text_upper for k in transfer_keywords):
        return ("anh_chuyen_khoan", 98.0)
    if any(c in filename for c in fn_transfer_clues):
        return ("anh_chuyen_khoan", 85.0)

    # 2. Kiểm tra Hợp đồng (Contract)
    if ("HỢP ĐỒNG" in header_text or "HOP DONG" in header_text) and \
       "ĐƠN ĐẶT HÀNG" not in header_text and "ĐƠN HÀNG" not in header_text:
        return ("hop_dong", 99.0)
        
    # 3. Kiểm tra Hóa đơn (Invoice)
    if "HÓA ĐƠN" in header_text or "HOÁ ĐƠN" in header_text or \
       "VAT INVOICE" in header_text or "INVOICE" in header_text:
        return ("hoa_don", 99.0)

    # 4. Kiểm tra Chứng từ / Đơn hàng (Voucher / Order / Receipt)
    if "ĐƠN ĐẶT HÀNG" in header_text or "ĐƠN HÀNG" in header_text or \
       "CHỨNG TỪ" in header_text or "PHIẾU" in header_text or \
       filename.startswith("so-") or "chung_tu" in filename:
        return ("chung_tu", 95.0)

    # Fallback dựa vào tên file và nội dung toàn văn
    if filename.startswith("hđ") or "hop_dong" in filename or "hợp đồng" in full_text.lower():
        return ("hop_dong", 80.0)
    if "hoadon" in filename or "invoice" in filename or "hóa đơn" in full_text.lower():
        return ("hoa_don", 80.0)
    if filename.startswith("so-") or "chung_tu" in filename or "chứng từ" in full_text.lower() or "đơn đặt hàng" in full_text.lower():
        return ("chung_tu", 80.0)
    if "chuyển khoản" in full_text.lower() or "chuyển tiền" in full_text.lower():
        return ("anh_chuyen_khoan", 75.0)

    return ("khac", 0.0)

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
    supported_exts = ('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp')
    
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
