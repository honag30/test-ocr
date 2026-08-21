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

INPUT_BASE_DIR = r"d:\AI\ocr\input"
DOC_BASE_DIR = r"d:\AI\ocr\doc"
OUTPUT_BASE_DIR = r"d:\AI\ocr\output"
PROCESSED_IMAGE_DIR = r"d:\AI\ocr\processed_images"

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

def save_ocr_output(original_filename: str, category_code: str, lines: list[str], confidences: list[float], eval_report: str = "", table_markdown: str = "", structured_tables: list = None) -> tuple[str, str]:
    """
    Lưu kết quả đọc OCR vào thư mục output:
    Mỗi tài liệu sẽ có 1 thư mục riêng trong danh mục:
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

    # 1. Lưu file TXT
    with open(out_path_txt, "w", encoding="utf-8") as out:
        out.write(f"=== KẾT QUẢ OCR: {original_filename} ===\n")
        out.write(f"Danh mục: {category_code}\n\n")
        
        for i, (l, conf) in enumerate(zip(lines, confidences), 1):
            out.write(format_line(i, conf, l) + "\n")
            
        if eval_report:
            out.write("\n" + eval_report + "\n")

    # 2. Lưu file JSON có cấu trúc kèm vị trí (Bounding Box)
    json_data = {
        "original_filename": original_filename,
        "category": category_code,
        "has_table": bool(structured_tables),
        "ocr_lines": [
            {
                "index": i,
                "text": l,
                "confidence": round(conf * 100.0 if conf is not None else 100.0, 2)
            }
            for i, (l, conf) in enumerate(zip(lines, confidences), 1)
        ],
        "tables": structured_tables or []
    }

    with open(out_path_json, "w", encoding="utf-8") as jout:
        json.dump(json_data, jout, ensure_ascii=False, indent=2)

    return out_path_txt, out_path_json

def ocr_single_image(image_path: str) -> tuple[list[str], list[float], str, list]:
    """
    Tiền xử lý và OCR ảnh bằng EasyOCR, đồng thời nhận diện cấu trúc Bảng và Bounding box.
    Trả về tuple: (lines, confidences, table_markdown, structured_tables)
    """
    print(f"\n[+] Đang tiền xử lý hình ảnh: {image_path}")
    processed_img = preprocess_image(image_path)
    
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    base_name = os.path.basename(image_path)
    out_img_path = os.path.join(PROCESSED_IMAGE_DIR, f"processed_{base_name}")
    cv2.imwrite(out_img_path, processed_img)
    print(f"-> Đã lưu ảnh đã qua tiền xử lý vào: '{out_img_path}'")

    reader = get_ocr_reader()
    print("[+] Đang chạy EasyOCR đọc văn bản & nhận diện Bảng...")
    raw_results = reader.readtext(
        processed_img,
        decoder='beamsearch',
        beamWidth=5,
        text_threshold=0.4,
        low_text=0.2,
        link_threshold=0.4,
        mag_ratio=1.5,
        paragraph=False
    )
    
    lines, confidences = group_results_by_line(raw_results, y_tolerance=20)
    
    # Nhận diện Bảng & Bounding Box
    table_matrix, is_table_found, structured_table = extract_table_from_raw_results(raw_results, image=processed_img)
    table_markdown = format_table_to_markdown(table_matrix) if is_table_found else ""
    structured_tables = [structured_table] if is_table_found else []
    
    if is_table_found:
        print("-> [✓] Phát hiện Bảng dữ liệu và vị trí (Bounding Box) trong hình ảnh!")

    return lines, confidences, table_markdown, structured_tables

def process_file_ocr(file_path: str) -> tuple[list[str], list[float], str, list]:
    """
    Đọc văn bản & nhận diện Bảng từ file (hỗ trợ cả PDF kỹ thuật số, PDF Scan và Ảnh).
    Trả về tuple: (lines, line_confidences, table_markdown, structured_tables)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
        tables_digital, structured_tables = extract_tables_from_pdf_digital(file_path)
        table_md = ""
        if tables_digital:
            md_list = [format_table_to_markdown(t) for t in tables_digital]
            table_md = "\n\n".join(md_list)
            print("-> [✓] Phát hiện Bảng kỹ thuật số từ file PDF!")

        if len(text.strip()) >= 30:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            confidences = [1.0] * len(lines)
            
            # Nếu chưa bóc được bảng từ pdfplumber, thử render trang PDF thành ảnh để chạy EasyOCR + OpenCV Table Detector
            if not table_md:
                pdf_images = render_pdf_to_images(file_path)
                if pdf_images:
                    reader = get_ocr_reader()
                    all_tables_md = []
                    structured_tables = []
                    for idx, img in enumerate(pdf_images):
                        raw = reader.readtext(img, text_threshold=0.4, low_text=0.2)
                        t_matrix, found, struct_t = extract_table_from_raw_results(raw, image=img)
                        if found:
                            all_tables_md.append(format_table_to_markdown(t_matrix))
                            struct_t['page'] = idx + 1
                            structured_tables.append(struct_t)
                    if all_tables_md:
                        table_md = "\n\n".join(all_tables_md)
                        print("-> [✓] Phát hiện Bảng từ hình ảnh trang PDF!")

            return lines, confidences, table_md, structured_tables
        else:
            print("-> PDF dạng scan/hình ảnh, cần chạy OCR ảnh...")
            pdf_images = render_pdf_to_images(file_path)
            if pdf_images:
                reader = get_ocr_reader()
                raw_results = reader.readtext(pdf_images[0], text_threshold=0.4, low_text=0.2)
                lines, confidences = group_results_by_line(raw_results, y_tolerance=20)
                t_matrix, found, struct_t = extract_table_from_raw_results(raw_results, image=pdf_images[0])
                table_md = format_table_to_markdown(t_matrix) if found else ""
                structured_tables = [struct_t] if found else []
                return lines, confidences, table_md, structured_tables
            return (["(PDF dạng scan cần sử dụng OCR ảnh)"], [0.5], "", [])
    elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
        return ocr_single_image(file_path)
    else:
        print(f"Định dạng file {ext} không hỗ trợ.")
        return [], [], "", []

def process_category_ocr(cat_key: str):
    """
    Thực hiện OCR cho các tài liệu thuộc 1 danh mục cụ thể và lưu vào thư mục output tương ứng.
    """
    cat_code, cat_name, folder_path = CATEGORIES[cat_key]
    
    print(f"\n" + "=" * 60)
    print(f"  THỰC HIỆN OCR CHO DANH MỤC: {cat_name.upper()}")
    print(f"  Thư mục tài liệu gốc: {folder_path}")
    print("=" * 60)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    supported_exts = ('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp')
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
        lines, confidences, table_md, struct_tables = process_file_ocr(file_path)
        
        metrics = evaluate_text_accuracy(lines, line_confidences=confidences)
        eval_report = print_evaluation_report(metrics, title=f"ĐÁNH GIÁ ĐỘ CHÍNH XÁC: {f}")
        
        print(f"\n--- KẾT QUẢ VĂN BẢN TRÍCH XUẤT ({f}) ---")
        for i, (l, conf) in enumerate(zip(lines, confidences), 1):
            print(format_line(i, conf, l))
            
        if table_md:
            print("\n--- BẢNG PHÁT HIỆN ĐƯỢC (TABLE STRUCTURE) ---")
            print(table_md)

        print("\n" + eval_report)
        
        out_txt, out_json = save_ocr_output(f, cat_code, lines, confidences, eval_report, table_md, struct_tables)
        saved_paths.append(out_txt)
        print(f"[✓] Đã lưu file TXT: '{out_txt}'")
        print(f"[✓] Đã lưu file JSON (Cấu trúc Bảng & Bounding Box): '{out_json}'")
        
    print(f"\n=> HOÀN THÀNH OCR! Tất cả kết quả đã lưu vào thư mục 'output/{cat_code}/'")

def process_custom_image():
    """
    Đọc OCR cho một file ảnh tùy chỉnh và lưu vào folder output tương ứng.
    """
    path = input("\nNhập đường dẫn file ảnh (mặc định 'image/1.png'): ").strip()
    if not path:
        path = "image/1.png"
        
    if not os.path.exists(path):
        print(f"File hoặc đường dẫn '{path}' không tồn tại!")
        return
        
    filename = os.path.basename(path)
    cat_code, conf = classify_document(path)
    if cat_code == "khac":
        cat_code = "khac"

    lines, confidences, table_md, struct_tables = ocr_single_image(path)
    metrics = evaluate_text_accuracy(lines, line_confidences=confidences)
    eval_report = print_evaluation_report(metrics, title=f"ĐÁNH GIÁ ĐỘ CHÍNH XÁC: {filename}")
    
    print(f"\n--- KẾT QUẢ ĐỌC VĂN BẢN: {filename} ---")
    for i, (l, conf_val) in enumerate(zip(lines, confidences), 1):
        print(format_line(i, conf_val, l))

    if table_md:
        print("\n--- BẢNG PHÁT HIỆN ĐƯỢC (TABLE STRUCTURE) ---")
        print(table_md)

    print("\n" + eval_report)
    
    out_txt, out_json = save_ocr_output(filename, cat_code, lines, confidences, eval_report, table_md, struct_tables)
    print(f"\n[✓] Đã lưu file TXT: '{out_txt}'")
    print(f"[✓] Đã lưu file JSON (Cấu trúc Bảng & Bounding Box): '{out_json}'")

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
    Tự động chạy OCR cho tất cả các tài liệu đã phân loại trong các thư mục con của 'doc'.
    """
    print("\n" + "=" * 60)
    print("  CHẠY OCR CHO TẤT CẢ CÁC TÀI LIỆU ĐÃ PHÂN LOẠI")
    print("=" * 60)
    
    total_processed = 0
    for key in ["1", "2", "3", "4"]:
        cat_code, cat_name, folder_path = CATEGORIES[key]
        if not os.path.exists(folder_path):
            continue
            
        supported_exts = ('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)]
        
        if not files:
            continue
            
        print(f"\n--- Đang xử lý danh mục [{cat_name}] ({len(files)} file) ---")
        for f in files:
            file_path = os.path.join(folder_path, f)
            lines, confidences, table_md, struct_tables = process_file_ocr(file_path)
            metrics = evaluate_text_accuracy(lines, line_confidences=confidences)
            eval_report = print_evaluation_report(metrics, title=f"ĐÁNH GIÁ ĐỘ CHÍNH XÁC: {f}")
            
            out_txt, out_json = save_ocr_output(f, cat_code, lines, confidences, eval_report, table_md, struct_tables)
            print(f"[✓] Đã OCR '{f}' -> TXT: '{out_txt}', JSON: '{out_json}'")
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

    supported_exts = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp")

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
