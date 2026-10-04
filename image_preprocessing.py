import os
import cv2
import numpy as np
from datasets import load_dataset

# 1. 처리 결과물 저장 디렉토리 생성
OUTPUT_DIR = "preprocessed_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Hugging Face 데이터셋 로드 
print("Hugging Face 데이터셋 로드 중...")
dataset = load_dataset("ethz/food101", split="train", streaming=True)

# 이상치 검출 임계값 설정
BRIGHTNESS_THRESHOLD = 50.0  # 평균 밝기 50 미만 시 어두운 이미지로 판단
MIN_OBJECT_RATIO = 0.05      # 전체 면적 대비 객체 비율 5% 미만 시 소형 객체로 판단

def is_too_dark(gray_img):
    """평균 밝기 기준 너무 어두운 이미지 필터링"""
    return np.mean(gray_img) < BRIGHTNESS_THRESHOLD

def is_object_too_small(gray_img):
    """이진화 및 윤곽선(Contour) 분석으로 객체 크기가 너무 작은 이미지 필터링"""
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    max_area = max(cv2.contourArea(c) for c in contours)
    total_area = gray_img.shape[0] * gray_img.shape[1]
    return (max_area / total_area) < MIN_OBJECT_RATIO

def augment_image(img):
    """데이터 증강: 좌우 반전, 회전, 밝기/대비 변화"""
    # 1) 좌우 반전
    flipped = cv2.flip(img, 1)
    # 2) 15도 회전
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rotated = cv2.warpAffine(flipped, matrix, (w, h))
    # 3) 밝기 및 대비 조정
    augmented = cv2.convertScaleAbs(rotated, alpha=1.1, beta=10)
    return augmented

saved_count = 0

for item in dataset:
    if saved_count >= 5:  # 처리된 이미지 5장 저장 시 종료
        break

    # PIL 이미지를 OpenCV BGR 포맷으로 변환
    pil_img = item["image"]
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 크기 조정 (224x224)
    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # 색상 변환 (Grayscale)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 이상치 탐지 및 필터링
    if is_too_dark(gray):
        print("필터링됨: 너무 어두운 이미지")
        continue
    if is_object_too_small(gray):
        print("필터링됨: 주요 객체 크기가 너무 작은 이미지")
        continue

    # 노이즈 제거 (Blur 필터 적용)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Normalization
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    # 데이터 증강 (좌우 반전, 회전, 색상 변화)
    processed_img = augment_image(normalized)

    # 결과 이미지 저장
    saved_count += 1
    save_path = os.path.join(OUTPUT_DIR, f"preprocessed_{saved_count}.png")
    cv2.imwrite(save_path, processed_img)
    print(f"저장 완료 ({saved_count}/5): {save_path}")

print("모든 이미지 전처리 및 이상치 필터링 완료")