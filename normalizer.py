import unicodedata
import re

def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt:
    - Unicode normalization (NFC)
    - Loại bỏ khoảng trắng thừa
    - Giữ nguyên các thông tin quan trọng: số tiền, ngày tháng, mã số thuế, số hợp đồng, v.v.
    """
    if not text:
        return ""

    # 1. Unicode Normalization -> Form NFC (dạng chuẩn của tiếng Việt)
    text = unicodedata.normalize('NFC', text)

    # 2. Xử lý các ký tự điều khiển không in được (trừ newline và tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. Chuẩn hóa khoảng trắng ngang (spaces, tabs) nhưng giữ ngắt dòng
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Thay thế tab và nhiều khoảng trắng liên tiếp bằng 1 space
        line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned_lines.append(line)

    # 4. Gộp các dòng rỗng thừa (tối đa 2 dòng rỗng liên tiếp)
    res = []
    empty_count = 0
    for line in cleaned_lines:
        if not line:
            empty_count += 1
            if empty_count <= 2:
                res.append("")
        else:
            empty_count = 0
            res.append(line)

    return "\n".join(res)


def is_usable_text(text: str, min_length: int = 20) -> bool:
    """
    Đánh giá chất lượng text trích xuất trực tiếp từ PDF:
    - Trả về False nếu text quá ngắn (< min_length)
    - Trả về False nếu dính lỗi font chữ (ví dụ: (cid:xxx), \\x00, \\ufffd)
    - Trả về False nếu tỷ lệ ký tự rác / ký tự không đọc được quá cao
    """
    if not text or len(text.strip()) < min_length:
        return False

    cleaned = text.strip()

    # 1. Kiểm tra lỗi font dạng PDF Font Mapping / CID (ví dụ: (cid:123), (cid:45))
    cid_matches = re.findall(r'\(cid:\d+\)', cleaned)
    if len(cid_matches) > 3 or (len(cleaned) > 0 and len(cid_matches) / max(1, len(cleaned.split())) > 0.2):
        return False

    # 2. Kiểm tra các ký tự replacement hoặc NULL byte
    if '\x00' in cleaned or '\ufffd' in cleaned:
        return False

    # 3. Tính tỷ lệ ký tự chữ/số/dấu tiếng Việt chuẩn so với tổng số ký tự
    valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                      "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
                      "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
                      " .,;:!?\"'()[]{}<>-+*/=_@#$%^&*~\n\r\t")

    valid_count = sum(1 for char in cleaned if char in valid_chars)
    valid_ratio = valid_count / len(cleaned)

    # Nếu tỷ lệ ký tự hợp lệ < 70%, đánh giá là text rác / lỗi encoding
    if valid_ratio < 0.70:
        return False

    # 4. Kiểm tra mật độ từ có nghĩa (tỷ lệ từ chứa ít nhất 1 chữ cái)
    words = cleaned.split()
    if not words:
        return False
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if len(alpha_words) / len(words) < 0.3:
        return False

    return True


def detect_headers_footers(page_texts: list[str], top_n_lines: int = 3, bottom_n_lines: int = 3, threshold: float = 0.5) -> tuple[set[str], set[str]]:
    """
    Phát hiện các dòng Header và Footer lặp đi lặp lại giữa các trang trong tài liệu nhiều trang.
    Trả về (headers, footers).
    """
    if len(page_texts) <= 1:
        return set(), set()

    total_pages = len(page_texts)
    header_candidates = {}
    footer_candidates = {}

    for page_text in page_texts:
        lines = [l.strip() for l in page_text.splitlines() if l.strip()]
        if not lines:
            continue

        # Top N lines -> Header candidates
        top_lines = lines[:min(top_n_lines, len(lines))]
        for l in top_lines:
            header_candidates[l] = header_candidates.get(l, 0) + 1

        # Bottom N lines -> Footer candidates
        bot_lines = lines[max(0, len(lines) - bottom_n_lines):]
        for l in bot_lines:
            footer_candidates[l] = footer_candidates.get(l, 0) + 1

    headers = {line for line, count in header_candidates.items() if count / total_pages >= threshold}
    footers = {line for line, count in footer_candidates.items() if count / total_pages >= threshold}

    return headers, footers
