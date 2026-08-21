import re

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Tính khoảng cách Levenshtein giữa 2 chuỗi ký tự.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Tính Tỷ lệ Lỗi Ký tự (Character Error Rate - CER).
    CER = Levenshtein_Distance(pred, truth) / len(truth)
    """
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    dist = levenshtein_distance(predicted, ground_truth)
    return min(1.0, dist / max(1, len(ground_truth)))

def calculate_wer(predicted: str, ground_truth: str) -> float:
    """
    Tính Tỷ lệ Lỗi Từ (Word Error Rate - WER).
    """
    words_pred = predicted.split()
    words_truth = ground_truth.split()
    if not words_truth:
        return 0.0 if not words_pred else 1.0
    dist = levenshtein_distance(words_pred, words_truth)
    return min(1.0, dist / max(1, len(words_truth)))

def evaluate_text_accuracy(predicted_lines: list[str], ground_truth_text: str = None, line_confidences: list[float] = None) -> dict:
    """
    Đánh giá độ chính xác tổng hợp của kết quả OCR:
    1. Trả về Độ tin cậy OCR trung bình (Confidence Score) từ Model.
    2. Nếu có Ground Truth (Văn bản đối chiếu): Tính CER, WER và Accuracy %.
    """
    full_pred = " ".join(predicted_lines).strip()
    
    metrics = {}
    
    # 1. Đánh giá Confidence Score từ model EasyOCR
    if line_confidences:
        avg_conf = sum(line_confidences) / len(line_confidences) if line_confidences else 0.0
        high_conf_ratio = sum(1 for c in line_confidences if c >= 0.8) / max(1, len(line_confidences)) * 100
        metrics['avg_confidence'] = round(avg_conf * 100, 2)
        metrics['high_confidence_ratio'] = round(high_conf_ratio, 2)
    else:
        metrics['avg_confidence'] = None
        metrics['high_confidence_ratio'] = None

    # 2. Đánh giá theo Ground Truth nếu có
    if ground_truth_text:
        cer = calculate_cer(full_pred, ground_truth_text.strip())
        wer = calculate_wer(full_pred, ground_truth_text.strip())
        accuracy_char = round((1.0 - cer) * 100, 2)
        accuracy_word = round((1.0 - wer) * 100, 2)
        
        metrics['cer'] = round(cer * 100, 2)
        metrics['wer'] = round(wer * 100, 2)
        metrics['char_accuracy'] = max(0.0, accuracy_char)
        metrics['word_accuracy'] = max(0.0, accuracy_word)
    
    return metrics

def print_evaluation_report(metrics: dict, title: str = "ĐÁNH GIÁ ĐỘ CHÍNH XÁC OCR") -> str:
    """
    In và tạo chuỗi báo cáo đánh giá độ chính xác.
    """
    report = []
    report.append("=" * 60)
    report.append(f"  {title.upper()}")
    report.append("=" * 60)
    
    if metrics.get('avg_confidence') is not None:
        report.append(f"🔹 Độ tin cậy OCR trung bình (Model Confidence): {metrics['avg_confidence']}%")
        report.append(f"🔹 Tỷ lệ dòng độ tin cậy cao (≥ 80%):         {metrics['high_confidence_ratio']}%")
        
        conf = metrics['avg_confidence']
        if conf >= 90:
            rating = "XUẤT SẮC (Chữ rất rõ ràng)"
        elif conf >= 75:
            rating = "TỐT (Đọc chính xác hầu hết)"
        elif conf >= 55:
            rating = "TRUNG BÌNH (Có một số nét chữ bị mờ)"
        else:
            rating = "THẤP (Ảnh bị mờ, nhiễu hoặc mất nét)"
        report.append(f"🔹 Đánh giá chất lượng đọc:                  {rating}")

    if metrics.get('char_accuracy') is not None:
        report.append("-" * 60)
        report.append("📊 ĐỐI CHIẾU VĂN BẢN CHUẨN (GROUND TRUTH):")
        report.append(f"  - Độ chính xác ký tự (Character Accuracy):  {metrics['char_accuracy']}%")
        report.append(f"  - Độ chính xác từ (Word Accuracy):          {metrics['word_accuracy']}%")
        report.append(f"  - Tỷ lệ lỗi ký tự (CER):                     {metrics['cer']}%")
        report.append(f"  - Tỷ lệ lỗi từ (WER):                        {metrics['wer']}%")
        
    report.append("=" * 60)
    return "\n".join(report)
