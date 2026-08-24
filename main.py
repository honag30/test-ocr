import os
import sys
import json
import cv2
import pypdf

from image_processor import preprocess_image
from ocr_utils import group_results_by_line
from document_classifier import organize_documents, get_ocr_reader, extract_text_from_pdf, classify_document
from evaluate_accuracy import evaluate_text_accuracy, print_evaluation_report
from table_detector import (
    extract_table_from_raw_results,
    extract_tables_from_pdf_digital,
    render_pdf_to_images,
    format_table_to_markdown
)

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_BASE_DIR = os.path.join(BASE_DIR, "input")
DOC_BASE_DIR = os.path.join(BASE_DIR, "doc")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "output")
PROCESSED_IMAGE_DIR = os.path.join(BASE_DIR, "processed_images")

CATEGORIES = {
    "1": ("hop_dong", "Hợp đồng", os.path.join(DOC_BASE_DIR, "hop_dong")),
    "2": ("hoa_don", "Hóa đơn", os.path.join(DOC_BASE_DIR, "hoa_don")),
    "3": ("chung_tu", "Chứng từ", os.path.join(DOC_BASE_DIR, "chung_tu")),
    "4": ("anh_chuyen_khoan", "Ảnh chuyển khoản", os.path.join(DOC_BASE_DIR, "anh_chuyen_khoan")),
}

def format_line(index: int, conf: float, text: str) -> str:
    """
    Định dạng dòng kết quả theo chuẩn gọn: [### - %%] Nội dung dòng
    """
    conf_val = conf * 100.0 if conf is not None else 100.0
    return f"[{index:03d} - {conf_val:.1f}%] {text}"

def save_ocr_output(original_filename: str, category_code: str, doc_result: dict, eval_report: str = "") -> tuple[str, str]:
    """
    Lưu kết quả đọc/trích xuất tài liệu vào thư mục output theo đúng cấu trúc chuẩn:
    output/{danh_mục}/{tên_file}/[OCR] - {tên_file}.txt
    output/{danh_mục}/{tên_file}/[OCR] - {tên_file}.json
    """
    base_name = os.path.splitext(original_filename)[0]
    doc_folder = os.path.join(OUTPUT_BASE_DIR, category_code, base_name)
    os.makedirs(doc_folder, exist_ok=True)

    out_filename_txt = f"[OCR] - {base_name}.txt"
    out_filename_json = f"[OCR] - {base_name}.json"
    
    out_path_txt = os.path.join(doc_folder, out_filename_txt)
    out_path_json = os.path.join(doc_folder, out_filename_json)

    full_text = doc_result.get("full_text", "")
    lines = [l for l in full_text.splitlines() if l.strip()]

    # 1. Lưu file TXT
    with open(out_path_txt, "w", encoding="utf-8") as out:
        out.write(f"=== KẾT QUẢ TRÍCH XUẤT TÀI LIỆU: {original_filename} ===\n")
        out.write(f"Danh mục: {category_code}\n")
        out.write(f"Định dạng nguồn (Source Type): {doc_result.get('source_type')}\n")
        out.write(f"Phương pháp trích xuất (Method): {doc_result.get('extraction_method')}\n\n")
        
        out.write("--- NỘI DUNG VĂN BẢN ---\n")
        out.write(full_text + "\n\n")

        tables = doc_result.get("tables", [])
        if tables:
            out.write("--- CẤU TRÚC BẢNG TRÍCH XUẤT ---\n")
            for t_idx, tbl in enumerate(tables, 1):
                out.write(f"\n[Bảng {t_idx} - Trang {tbl.get('page', 1)}]\n")
                out.write(tbl.get("markdown", "") + "\n")
            
        if eval_report:
            out.write("\n" + eval_report + "\n")

    # 2. Lưu file JSON cấu trúc mở rộng (Tương thích cả API cũ lẫn cấu trúc mở rộng mới)
    ocr_lines_legacy = []
    line_idx = 1
    for pg in doc_result.get("pages", []):
        pg_lines = pg.get("lines", [])
        pg_confs = pg.get("confidences", [])
        for i, l in enumerate(pg_lines):
            conf_val = pg_confs[i] * 100.0 if (pg_confs and i < len(pg_confs) and pg_confs[i] is not None) else None
            ocr_lines_legacy.append({
                "index": line_idx,
                "page": pg.get("page", 1),
                "text": l,
                "confidence": conf_val
            })
            line_idx += 1

    json_data = {
        "original_filename": original_filename,
        "category": category_code,
        "document_type": doc_result.get("document_type"),
        "source_type": doc_result.get("source_type"),
        "extraction_method": doc_result.get("extraction_method"),
        "has_table": doc_result.get("has_table", False),
        "total_pages": doc_result.get("total_pages", 1),
        "pages": doc_result.get("pages", []),
        "full_text": full_text,
        "tables": doc_result.get("tables", []),
        "ocr_lines": ocr_lines_legacy
    }

    with open(out_path_json, "w", encoding="utf-8") as jout:
        json.dump(json_data, jout, ensure_ascii=False, indent=2)

    return out_path_txt, out_path_json

def process_file_ocr(file_path: str) -> dict:
    """
    Đọc trích xuất văn bản & nhận diện Bảng từ file (hỗ trợ PDF text/scan/mixed, DOCX, XLSX và Ảnh).
    """
    from document_reader import read_document
    return read_document(file_path)


def process_category_ocr(cat_key: str):
    """
    Thực hiện đọc & trích xuất cho các tài liệu thuộc 1 danh mục cụ thể và lưu vào thư mục output.
    """
    cat_code, cat_name, folder_path = CATEGORIES[cat_key]
    
    print(f"\n" + "=" * 60)
    print(f"  TRÍCH XUẤT TÀI LIỆU DANH MỤC: {cat_name.upper()}")
    print(f"  Thư mục tài liệu gốc: {folder_path}")
    print("=" * 60)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    supported_exts = ('.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg', '.webp', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)]
    
    if not files:
        print(f"⚠️  Không tìm thấy tài liệu nào trong thư mục '{folder_path}'.")
        print("💡 Gợi ý: Bạn có thể chọn chức năng [7] trên menu để phân loại tự động từ thư mục 'doc'.")
        return

    print(f"Phát hiện {len(files)} tài liệu:")
    for idx, f in enumerate(files, 1):
        print(f"  [{idx}] {f}")
    print(f"  [A] Đọc tất cả {len(files)} file")
    
    choice = input("\nNhập số thứ tự file cần đọc (hoặc chọn 'A' để đọc tất cả): ").strip()
    
    selected_files = []
    if choice.lower() == 'a':
        selected_files = files
    elif choice.isdigit() and 1 <= int(choice) <= len(files):
        selected_files = [files[int(choice) - 1]]
    else:
        print("Lựa chọn không hợp lệ!")
        return
        
    saved_paths = []
    for f in selected_files:
        file_path = os.path.join(folder_path, f)
        print(f"\n>>> Đang xử lý file: {f}")
        doc_result = process_file_ocr(file_path)
        
        full_text = doc_result.get("full_text", "")
        lines = [l for l in full_text.splitlines() if l.strip()]
        
        all_confs = []
        for pg in doc_result.get("pages", []):
            confs = pg.get("confidences", [])
            if confs:
                all_confs.extend(confs)

        metrics = evaluate_text_accuracy(lines, line_confidences=all_confs if all_confs else None)
        eval_report = print_evaluation_report(metrics, title=f"ĐÁNH GIÁ TRÍCH XUẤT: {f}")
        
        print(f"\n--- KẾT QUẢ VĂN BẢN TRÍCH XUẤT ({f}) ---")
        print(full_text)
            
        tables = doc_result.get("tables", [])
        if tables:
            print("\n--- BẢNG PHÁT HIỆN ĐƯỢC (TABLE STRUCTURE) ---")
            for t_idx, tbl in enumerate(tables, 1):
                print(f"\n[Bảng {t_idx} - Trang {tbl.get('page', 1)}]")
                print(tbl.get("markdown", ""))

        print("\n" + eval_report)
        
        out_txt, out_json = save_ocr_output(f, cat_code, doc_result, eval_report)
        saved_paths.append(out_txt)
        print(f"[✓] Đã lưu file TXT: '{out_txt}'")
        print(f"[✓] Đã lưu file JSON (Cấu trúc mở rộng & Bảng): '{out_json}'")
        
    print(f"\n=> HOÀN THÀNH TRÍCH XUẤT! Tất cả kết quả đã lưu vào thư mục 'output/{cat_code}/'")

def process_custom_image():
    """
    Đọc trích xuất một file bất kỳ (PDF, DOCX, XLSX, Ảnh) và lưu vào folder output tương ứng.
    """
    path = input("\nNhập đường dẫn file (ví dụ 'input/6.png' hoặc chọn Enter để dùng mặc định): ").strip()
    if not path:
        if os.path.exists("input/6.png"):
            path = "input/6.png"
        elif os.path.exists("image/1.png"):
            path = "image/1.png"
        else:
            path = "input"

    if not os.path.exists(path):
        print(f"⚠️ File hoặc đường dẫn '{path}' không tồn tại!")
        return
        
    filename = os.path.basename(path)
    cat_code, conf = classify_document(path)
    if cat_code == "khac":
        cat_code = "khac"

    doc_result = process_file_ocr(path)
    full_text = doc_result.get("full_text", "")
    lines = [l for l in full_text.splitlines() if l.strip()]

    all_confs = []
    for pg in doc_result.get("pages", []):
        confs = pg.get("confidences", [])
        if confs:
            all_confs.extend(confs)

    metrics = evaluate_text_accuracy(lines, line_confidences=all_confs if all_confs else None)
    eval_report = print_evaluation_report(metrics, title=f"ĐÁNH GIÁ TRÍCH XUẤT: {filename}")
    
    print(f"\n--- KẾT QUẢ ĐỌC VĂN BẢN: {filename} ---")
    print(full_text)

    tables = doc_result.get("tables", [])
    if tables:
        print("\n--- BẢNG PHÁT HIỆN ĐƯỢC (TABLE STRUCTURE) ---")
        for t_idx, tbl in enumerate(tables, 1):
            print(f"\n[Bảng {t_idx} - Trang {tbl.get('page', 1)}]")
            print(tbl.get("markdown", ""))

    print("\n" + eval_report)
    
    out_txt, out_json = save_ocr_output(filename, cat_code, doc_result, eval_report)
    print(f"\n[✓] Đã lưu file TXT: '{out_txt}'")
    print(f"[✓] Đã lưu file JSON: '{out_json}'")

def run_ground_truth_evaluation():
    """
    So sánh đối chiếu kết quả OCR với Văn bản chuẩn (Ground Truth) để tính tỷ lệ lỗi CER / WER / Accuracy %.
    """
    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ ĐỘ CHÍNH XÁC VỚI VĂN BẢN CHUẨN (GROUND TRUTH)")
    print("=" * 60)
    
    pred_path = input("Nhập đường dẫn file kết quả OCR cần đối chiếu: ").strip()
    if not pred_path or not os.path.exists(pred_path):
        print(f"File '{pred_path}' không tồn tại!")
        return
        
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_text = f.read()
        
    print("\nNhập/Dán văn bản chuẩn (Ground Truth) để đối chiếu (kết thúc bằng ấn Enter 2 lần liên tiếp):")
    gt_lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            if not line:
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                gt_lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
            
    gt_text = "\n".join(gt_lines).strip()
    if not gt_text:
        print("Chưa nhập văn bản chuẩn!")
        return
        
    metrics = evaluate_text_accuracy(pred_text.split('\n'), ground_truth_text=gt_text)
    report = print_evaluation_report(metrics, title="KẾT QUẢ ĐỐI CHIẾU VĂN BẢN CHUẨN")
    print("\n" + report)

def run_all_categories_ocr():
    """
    Tự động kiểm tra thư mục 'input' & 'doc', phân loại các tài liệu chưa phân loại và chạy trích xuất toàn bộ.
    """
    print("\n" + "=" * 60)
    print("  CHẠY TRÍCH XUẤT CHO TẤT CẢ CÁC TÀI LIỆU")
    print("=" * 60)

    # 1. Kiểm tra và tự động phân loại các file trong thư mục input nếu có
    if os.path.exists(INPUT_BASE_DIR):
        supported_exts = ('.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg', '.webp', '.bmp')
        input_files = [f for f in os.listdir(INPUT_BASE_DIR) if os.path.isfile(os.path.join(INPUT_BASE_DIR, f)) and f.lower().endswith(supported_exts)]
        if input_files:
            print(f"[Input] Phát hiện {len(input_files)} tài liệu trong thư mục '{INPUT_BASE_DIR}'. Tiến hành phân loại vào 'doc/'...")
            organize_documents(INPUT_BASE_DIR, DOC_BASE_DIR)

    
    total_processed = 0
    for key in ["1", "2", "3", "4"]:
        cat_code, cat_name, folder_path = CATEGORIES[key]
        if not os.path.exists(folder_path):
            continue
            
        supported_exts = ('.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg', '.webp', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)]
        
        if not files:
            continue
            
        print(f"\n--- Đang xử lý danh mục [{cat_name}] ({len(files)} file) ---")
        for f in files:
            file_path = os.path.join(folder_path, f)
            doc_result = process_file_ocr(file_path)

            full_text = doc_result.get("full_text", "")
            lines = [l for l in full_text.splitlines() if l.strip()]

            all_confs = []
            for pg in doc_result.get("pages", []):
                confs = pg.get("confidences", [])
                if confs:
                    all_confs.extend(confs)

            metrics = evaluate_text_accuracy(lines, line_confidences=all_confs if all_confs else None)
            eval_report = print_evaluation_report(metrics, title=f"ĐÁNH GIÁ TRÍCH XUẤT: {f}")
            
            out_txt, out_json = save_ocr_output(f, cat_code, doc_result, eval_report)
            print(f"[✓] Đã xử lý '{f}' -> TXT: '{out_txt}', JSON: '{out_json}'")
            total_processed += 1
            
    print(f"\n=> TỔNG CỘNG ĐÃ OCR HOÀN TẤT {total_processed} FILE! Kết quả được lưu tại thư mục 'output/'.")

MENU_ITEMS = {
    "1": ("📄 Đọc OCR - Hợp đồng", process_category_ocr),
    "2": ("🧾 Đọc OCR - Hóa đơn", process_category_ocr),
    "3": ("📑 Đọc OCR - Chứng từ", process_category_ocr),
    "4": ("💸 Đọc OCR - Ảnh chuyển khoản", process_category_ocr),
    "5": ("🚀 Đọc OCR - Tất cả tài liệu", lambda: run_all_categories_ocr()),
    "6": ("🖼️ Đọc OCR - File ảnh bất kỳ", process_custom_image),
    "7": ("⚡ Tự động phân loại", None),
    "8": ("📊 Đánh giá CER/WER", run_ground_truth_evaluation),
    "0": ("❌ Thoát", None),
}

def show_menu():
    print("\n" + "=" * 60)
    print("      CHƯƠNG TRÌNH PHÂN LOẠI & ĐỌC OCR TÀI LIỆU")
    print("=" * 60)
    print(" 1. 📄 Đọc OCR - Hợp đồng           (doc/hop_dong)")
    print(" 2. 🧾 Đọc OCR - Hóa đơn            (doc/hoa_don)")
    print(" 3. 📑 Đọc OCR - Chứng từ           (doc/chung_tu)")
    print(" 4. 💸 Đọc OCR - Ảnh chuyển khoản   (doc/anh_chuyen_khoan)")
    print(" 5. 🚀 Đọc OCR - Tất cả các loại tài liệu")
    print(" 6. 🖼️ Đọc OCR - File ảnh bất kỳ")
    print(" 7. ⚡ Tự động Phân loại & Chuyển file")
    print(" 8. 📊 Đánh giá độ chính xác (CER/WER)")
    print(" 0. ❌ Thoát")
    print("=" * 60)

def auto_classify():
    """Tự động phân loại file từ input/doc rồi OCR nếu người dùng đồng ý."""

    print("\n" + "=" * 60)
    print("TỰ ĐỘNG PHÂN LOẠI FILE")
    print("=" * 60)

    supported_exts = (".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def get_files(folder):
        if not os.path.exists(folder):
            return []
        return [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith(supported_exts)
        ]

    source_dir = INPUT_BASE_DIR if get_files(INPUT_BASE_DIR) else DOC_BASE_DIR

    if source_dir == INPUT_BASE_DIR:
        print(f"Phát hiện {len(get_files(source_dir))} file trong 'input'.")
    else:
        print("Không có file trong 'input', kiểm tra trực tiếp trong 'doc'.")

    moved = organize_documents(source_dir, DOC_BASE_DIR)

    if moved:
        answer = input(
            "\nOCR tất cả file vừa phân loại? (Y/n): "
        ).strip().lower()

        if answer in ("", "y", "yes", "có", "co"):
            run_all_categories_ocr()

def main_menu():
    while True:
        show_menu()

        try:
            choice = input("Nhập lựa chọn (0-8): ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "0":
            print("\nCảm ơn bạn đã sử dụng chương trình! Tạm biệt.")
            break

        if choice in ("1", "2", "3", "4"):
            process_category_ocr(choice)
            continue

        if choice == "7":
            auto_classify()
            continue

        action = MENU_ITEMS.get(choice)

        if action and action[1]:
            action[1]()
        else:
            print("\n⚠️ Lựa chọn không hợp lệ, vui lòng thử lại!")

if __name__ == "__main__":
    main_menu()
