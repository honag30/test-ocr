# test-ocr

Công cụ OCR & trích xuất thông tin tự động từ tài liệu dạng PDF, DOCX, XLSX và Ảnh, tập trung vào hợp đồng, hóa đơn, chứng từ kế toán tiếng Việt.

## Tính năng chính

- **Ưu tiên Native Text**: Đọc trực tiếp text nhúng từ PDF kỹ thuật số (không cần OCR), chỉ dùng EasyOCR khi gặp PDF scan hoặc ảnh.
- **Table Extraction chính xác**: Tái tạo bảng 2D (hàng × cột) từ tọa độ XY của văn bản, hỗ trợ bảng có/không có đường kẻ, gộp multiline cell.
- **Loại bỏ Paragraph khỏi Bảng**: Tự động phân vùng và loại bỏ đoạn văn bản thường (ĐIỀU 1, Bên A, Bên B, điều khoản pháp lý) không đưa vào cell bảng.
- **Validation Engine tài chính**: Kiểm tra tự động `Subtotal + VAT ≈ Grand Total`, `Price - Discount ≈ Subtotal` và trả về trạng thái `"pass"`, `"warning"`, hoặc `"not_checkable"`.
- **Hỗ trợ đa định dạng**: `.pdf`, `.docx`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.
- **Output chuẩn hóa**: Xuất đồng thời file `.txt` và `.json` có cấu trúc đầy đủ theo từng danh mục tài liệu.
- **Quality Gate tự động**: Bộ kiểm thử tự động (`test_quality_gate.py`) xác minh 10 tiêu chí chất lượng bảng và chạy regression test trên toàn bộ tài liệu mẫu.

---

## Kiến trúc

```
test-ocr/
├── main.py                  # CLI menu chính: chọn danh mục, đọc file, lưu output
├── document_reader.py       # Pipeline trích xuất text + bảng từ mọi định dạng file
├── table_detector.py        # Thuật toán phát hiện & tái tạo bảng (spatial XY clustering)
├── document_classifier.py   # Phân loại loại tài liệu (hợp đồng, hóa đơn, chứng từ, ảnh)
├── image_processor.py       # Tiền xử lý ảnh: deskew, threshold, noise removal
├── ocr_utils.py             # Nhóm kết quả EasyOCR theo dòng
├── normalizer.py            # Chuẩn hóa text tiếng Việt, kiểm tra usable text
├── evaluate_accuracy.py     # Đánh giá độ chính xác nhận diện ký tự
├── test_quality_gate.py     # Kiểm thử Quality Gate baseline + Regression Test
├── test_pipeline.py         # Test pipeline xử lý cơ bản
│
├── doc/                     # Thư mục tài liệu đầu vào (phân theo danh mục)
│   ├── hop_dong/            # Hợp đồng
│   ├── hoa_don/             # Hóa đơn
│   ├── chung_tu/            # Chứng từ kế toán
│   └── anh_chuyen_khoan/    # Ảnh chuyển khoản
│
├── output/                  # Thư mục lưu kết quả OCR (tự động tạo)
│   └── {danh_muc}/{ten_file}/
│       ├── [OCR] - {ten_file}.txt
│       └── [OCR] - {ten_file}.json
│
└── input/                   # Thư mục nhập liệu thủ công (tùy chọn)
```

---

## Yêu cầu

- Python 3.10+
- Các thư viện chính (cài qua `pip`):
  ```
  pymupdf
  pdfplumber
  pypdf
  easyocr
  torch
  opencv-python
  python-docx
  openpyxl
  numpy
  ```

---

## Cài đặt & Chạy

```bash
# 1. Clone repository
git clone <repository-url>
cd test-ocr

# 2. Cài đặt thư viện
pip install pymupdf pdfplumber pypdf easyocr torch opencv-python python-docx openpyxl numpy

# 3. Chạy menu chính
python main.py
```

---

## Sử dụng

### Menu tương tác CLI

```
======================================================
  [1] Đọc & Trích xuất: Hợp đồng
  [2] Đọc & Trích xuất: Hóa đơn
  [3] Đọc & Trích xuất: Chứng từ
  [4] Đọc & Trích xuất: Ảnh chuyển khoản
  [7] Phân loại tự động tài liệu mới từ thư mục input/
  [0] Thoát
======================================================
```

### Trực tiếp trong code

```python
from document_reader import read_document

result = read_document("doc/hop_dong/HĐ tên miền verco.vn (1).pdf")

print(result["source_type"])      # "pdf_text" | "pdf_scan" | "image" | "docx" | "xlsx"
print(result["extraction_method"])# "native_text" | "ocr" | "mixed" | "table_extraction"
print(result["has_table"])         # True / False
print(result["full_text"])         # Toàn bộ nội dung văn bản
print(result["tables"])            # Danh sách bảng trích xuất được
```

---

## Cấu trúc Output JSON

```json
{
  "original_filename": "HĐ tên miền verco.vn (1).pdf",
  "category": "hop_dong",
  "document_type": "contract",
  "source_type": "pdf_text",
  "extraction_method": "native_text",
  "has_table": true,
  "total_pages": 6,
  "full_text": "...",
  "pages": [
    {
      "page": 5,
      "text": "...",
      "lines": ["..."],
      "confidences": [null]
    }
  ],
  "tables": [
    {
      "table_index": 0,
      "page": 5,
      "detection_method": "native_pdf_spatial",
      "table_confidence": 0.98,
      "headers": ["STT", "Gói dịch vụ theo đơn hàng số", "Thời gian", "Giá (VND)", "..."],
      "columns": [
        {"index": 0, "name": "STT", "x_start": 40.7, "x_end": 60.0},
        {"index": 1, "name": "Gói dịch vụ", "x_start": 60.0, "x_end": 195.0}
      ],
      "rows": [
        {
          "row_index": 1,
          "cells": [
            {"column_index": 0, "column_name": "STT", "text": "1"},
            {"column_index": 1, "column_name": "Gói dịch vụ", "text": "Lệ phí đăng ký .vn verco.vn"}
          ]
        }
      ],
      "matrix": [
        ["STT", "Gói dịch vụ", "Thời gian", "Giá (VND)", "Giảm giá", "Thành tiền", "Thuế VAT", "Thành tiền sau thuế"],
        ["1", "Lệ phí đăng ký .vn verco.vn", "1 Lần", "100,000", "0", "100,000", "0", "100,000"],
        ["", "Cộng tiền các khoản (VND)", "", "599,000", "0", "599,000", "11,920", "610,920"]
      ],
      "markdown": "| STT | Gói dịch vụ | Thời gian | ... |",
      "validation": {
        "status": "pass",
        "issues": []
      }
    }
  ],
  "ocr_lines": [
    {"index": 1, "page": 1, "text": "...", "confidence": 99.5}
  ]
}
```

---

## Chạy Quality Gate & Regression Test

```bash
python test_quality_gate.py
```

**Kết quả mẫu (đạt chuẩn):**

```
======================================================================
  [QUALITY GATE] KIỂM TRA TỰ ĐỘNG FILE BASELINE: 'HĐ tên miền verco.vn (1).pdf'
======================================================================
  [✓] Check 1: has_table == True và phát hiện được Bảng
  [✓] Check 2: Không chứa đoạn văn bản (ĐIỀU 1, Bên A, Bên B...) trong Table cell
  [✓] Check 3: Matrix 2D hợp lệ (6 rows x 12 cols)
  [✓] Check 4: Số lượng cột hợp lý (12 cột)
  [✓] Check 5: Header và thứ tự cột khớp thực tế
  [✓] Check 6: Trích xuất đầy đủ số tiền quan trọng (599,000 / 11,920 / 610,920)
  [✓] Check 7: Trích xuất đúng ngày tháng (02-08-2026, 02-08-2027)
  [✓] Check 8: Hỗ trợ gộp Multiline cell
  [✓] Check 9: Validation Engine PASS (Subtotal + VAT == Grand Total)
  [✓] Check 10: Chất lượng OCR Text bảo toàn nguyên vẹn (23,858 ký tự)
----------------------------------------------------------------------
KẾT QUẢ QUALITY GATE BASELINE: 10/10 CHECKS PASSED

======================================================================
KẾT QUẢ REGRESSION TEST: 12/12 FILES PASSED (Failed: 0)
======================================================================
```

---

## Các biến môi trường

| Biến | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `DEBUG_TABLE` | `0` | Bật log chi tiết thuật toán phát hiện bảng (`"1"` để bật) |

```bash
# Bật debug log bảng
DEBUG_TABLE=1 python main.py
```

---

## Ghi chú

- **PDF kỹ thuật số (có text layer)**: Ưu tiên đọc native không cần OCR, độ chính xác ~99%.
- **PDF scan / ảnh**: Tự động fallback sang EasyOCR (`vi`, `en`). Tốc độ chậm hơn, yêu cầu CPU hoặc GPU.
- **Bảng không có đường kẻ (borderless tables)**: Phát hiện dựa trên sự dóng hàng/cột không gian (spatial X/Y alignment).
- **Tương thích ngược**: Toàn bộ key JSON cũ (`original_filename`, `category`, `has_table`, `pages`, `full_text`, `tables`, `ocr_lines`) được giữ nguyên.