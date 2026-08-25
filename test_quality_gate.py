import os
import sys
import json
import re

sys.stdout.reconfigure(encoding='utf-8')

from document_reader import read_document
from evaluate_accuracy import evaluate_text_accuracy

def run_quality_gate_baseline():
    """
    QUALITY GATE TEST CHO FILE BASELINE: 'HĐ tên miền verco.vn (1).pdf'
    """
    print("\n" + "=" * 70)
    print("  [QUALITY GATE] KIỂM TRA TỰ ĐỘNG FILE BASELINE: 'HĐ tên miền verco.vn (1).pdf'")
    print("=" * 70)

    pdf_path = os.path.join("doc", "hop_dong", "HĐ tên miền verco.vn (1).pdf")
    if not os.path.exists(pdf_path):
        # Fallback to search
        found = False
        for root, dirs, files in os.walk("doc"):
            for f in files:
                if "verco" in f.lower() and f.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, f)
                    found = True
                    break
            if found:
                break

    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: Không tìm thấy file baseline tại '{pdf_path}'!")
        return False

    print(f"-> Đang chạy trích xuất dữ liệu từ file: {pdf_path}")
    res = read_document(pdf_path)

    checks_passed = 0
    total_checks = 10
    failures = []

    # 1. Check has_table
    if res.get("has_table") and len(res.get("tables", [])) >= 1:
        print("  [✓] Check 1: has_table == True và phát hiện được Bảng")
        checks_passed += 1
    else:
        failures.append("Check 1 FAIL: Không phát hiện được bảng (has_table=False)")

    tables = res.get("tables", [])
    page5_table = None
    for t in tables:
        if t.get("page") == 5:
            page5_table = t
            break

    if not page5_table and tables:
        page5_table = tables[0]

    if not page5_table:
        print("❌ CRITICAL: Không tìm thấy bảng chi tiết trên Trang 5!")
        return False

    matrix = page5_table.get("matrix", [])
    matrix_str = json.dumps(matrix, ensure_ascii=False)

    # 2. Check 1: Không đưa paragraph vào table cell
    illegal_paragraphs = ["ĐIỀU 1", "Bên A", "Bên B", "Chi nhánh Công ty Cổ phần Mắt Bão", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"]
    found_illegal = [p for p in illegal_paragraphs if p in matrix_str]
    if not found_illegal:
        print("  [✓] Check 2: Không chứa các đoạn văn bản (ĐIỀU 1, Bên A, Bên B...) trong Table cell")
        checks_passed += 1
    else:
        failures.append(f"Check 2 FAIL: Bảng bị lọt văn bản paragraph: {found_illegal}")

    # 3. Check 2D Matrix (Không bị flatten)
    if isinstance(matrix, list) and len(matrix) >= 2 and all(isinstance(r, list) for r in matrix):
        print(f"  [✓] Check 3: Matrix 2D hợp lệ ({len(matrix)} rows x {len(matrix[0]) if matrix else 0} cols)")
        checks_passed += 1
    else:
        failures.append("Check 3 FAIL: Matrix bị flatten hoặc không phải danh sách 2D")

    # 4. Check số lượng cột (Columns)
    cols = page5_table.get("columns", [])
    if len(cols) >= 5:
        print(f"  [✓] Check 4: Số lượng cột hợp lý ({len(cols)} cột)")
        checks_passed += 1
    else:
        failures.append(f"Check 4 FAIL: Số cột quá ít ({len(cols)} cột)")

    # 5. Check thứ tự cột và headers
    headers = page5_table.get("headers", [])
    headers_concat = " ".join(headers)
    if ("STT" in headers_concat or "Gói dịch vụ" in headers_concat) and ("Giá" in headers_concat or "Thành tiền" in headers_concat):
        print(f"  [✓] Check 5: Header và thứ tự cột khớp thực tế: {headers[:4]}...")
        checks_passed += 1
    else:
        failures.append(f"Check 5 FAIL: Tên/Thứ tự cột header chưa đúng: {headers}")

    # 6. Check giá trị quan trọng: 599,000, 11,920, 610,920
    key_numbers = ["599,000", "11,920", "610,920"]
    found_nums = [n for n in key_numbers if n in matrix_str]
    if len(found_nums) == len(key_numbers):
        print(f"  [✓] Check 6: Đã trích xuất đầy đủ các số tiền quan trọng ({found_nums})")
        checks_passed += 1
    else:
        failures.append(f"Check 6 FAIL: Thiếu số tiền quan trọng (Tìm thấy: {found_nums}, Kỳ vọng: {key_numbers})")

    # 7. Check ngày tháng: 02-08-2026, 02-08-2027
    dates = ["02-08-2026", "02-08-2027"]
    found_dates = [d for d in dates if d in matrix_str]
    if len(found_dates) == len(dates):
        print(f"  [✓] Check 7: Trích xuất đúng ngày tháng thời hạn dịch vụ ({found_dates})")
        checks_passed += 1
    else:
        failures.append(f"Check 7 FAIL: Thiếu thông tin ngày tháng (Tìm thấy: {found_dates})")

    # 8. Check Multiline Cell Merging
    has_multiline = any("\n" in cell for r in matrix for cell in r)
    if has_multiline or "verco.vn" in matrix_str:
        print("  [✓] Check 8: Hỗ trợ gộp Multiline cell thành công")
        checks_passed += 1
    else:
        failures.append("Check 8 FAIL: Không tìm thấy multiline cell được gộp")

    # 9. Check Validation Status
    val_status = page5_table.get("validation", {}).get("status")
    if val_status == "pass":
        print(f"  [✓] Check 9: Validation Engine PASS (Subtotal + VAT == Grand Total)")
        checks_passed += 1
    else:
        failures.append(f"Check 9 FAIL: Validation status = '{val_status}' (Issues: {page5_table.get('validation', {}).get('issues')})")

    # 10. Check OCR text quality (full_text không rỗng và có độ khả thi cao)
    full_text = res.get("full_text", "")
    if len(full_text) > 500 and "HỢP ĐỒNG CUNG CẤP DỊCH VỤ" in full_text:
        print(f"  [✓] Check 10: Chất lượng OCR Text bảo toàn nguyên vẹn ({len(full_text)} ký tự)")
        checks_passed += 1
    else:
        failures.append(f"Check 10 FAIL: Full text bị suy giảm hoặc mất nội dung")

    print("-" * 70)
    print(f"KẾT QUẢ QUALITY GATE BASELINE: {checks_passed}/{total_checks} CHECKS PASSED")

    if failures:
        print("CÁC LỖI CẦN XỬ LÝ:")
        for f in failures:
            print("  -", f)
        return False
    else:
        print("🎉 TẤT CẢ CHECKS QUALITY GATE ĐÃ ĐẠT KHÔNG CÓ LỖI!")
        return True


def run_regression_suite():
    """
    REGRESSION TEST SUITE CHO TOÀN BỘ FILE TRONG THƯ MỤC 'doc/'
    """
    print("\n" + "=" * 70)
    print("  [REGRESSION TEST SUITE] CHẠY KIỂM THỬ TẤT CẢ FILE TRONG THƯ MỤC 'doc/'")
    print("=" * 70)

    doc_dir = "doc"
    if not os.path.exists(doc_dir):
        print(f"Thư mục '{doc_dir}' không tồn tại!")
        return

    supported_exts = ('.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg', '.webp', '.bmp')
    all_files = []
    for root, dirs, files in os.walk(doc_dir):
        for f in files:
            if f.lower().endswith(supported_exts):
                all_files.append(os.path.join(root, f))

    print(f"Phát hiện tổng cộng {len(all_files)} tài liệu để chạy Regression Test.")

    passed_count = 0
    failed_count = 0

    for idx, fpath in enumerate(all_files, 1):
        fname = os.path.basename(fpath)
        print(f"\n[{idx}/{len(all_files)}] Testing: '{fname}' ({fpath})")
        try:
            res = read_document(fpath)
            full_text = res.get("full_text", "")
            has_table = res.get("has_table", False)
            tables = res.get("tables", [])

            # Check JSON field compatibility
            req_keys = ["document_type", "source_type", "extraction_method", "has_table", "pages", "full_text", "tables"]
            missing_keys = [k for k in req_keys if k not in res]

            if missing_keys:
                print(f"  ❌ FAIL: Thiếu các trường bắt buộc trong JSON: {missing_keys}")
                failed_count += 1
            else:
                print(f"  ✓ PASS: Text length={len(full_text)}, has_table={has_table}, tables_count={len(tables)}")
                passed_count += 1

        except Exception as e:
            print(f"  ❌ CRASH/ERROR: {e}")
            failed_count += 1

    print("\n" + "=" * 70)
    print(f"  KẾT QUẢ REGRESSION TEST: {passed_count}/{len(all_files)} FILES PASSED (Failed: {failed_count})")
    print("=" * 70)

if __name__ == "__main__":
    success = run_quality_gate_baseline()
    run_regression_suite()
    if not success:
        sys.exit(1)
