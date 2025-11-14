from pathlib import Path

# ===== 설정 =====
LABEL_DIR = Path("/home/yeji/mk_add_experiments/dataset/target12/yolov8_preds/labels")  # 입력 폴더
OUT_DIR   = Path("/home/yeji/mk_add_experiments/dataset/target12/yolov8_preds/labels_no_conf")  # 출력 폴더

def remove_confidence_from_yolo_segmentation_labels(label_dir: Path, out_dir: Path):
    txt_files = list(label_dir.glob("*.txt"))
    if not txt_files:
        print(f"⚠️ No .txt files found in: {label_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()

            # segmentation 형식 예: class x1 y1 x2 y2 ... xn yn conf
            # → conf 제거 (마지막 값)
            if len(parts) > 3:
                # 신뢰도(conf)는 보통 마지막에 1개 float 값
                try:
                    float(parts[-1])
                    parts = parts[:-1]  # 마지막 값이 float이면 제거
                except ValueError:
                    pass  # 마지막 항목이 숫자가 아니면 그대로 둠

            new_lines.append(" ".join(parts))

        # 출력 파일 경로 (동일 이름)
        out_path = out_dir / txt_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        print(f"✅ Saved (no conf): {out_path.name}")

    print(f"\n💾 모든 segmentation 라벨에서 confidence 제거 완료.")
    print(f"→ 결과 저장 경로: {out_dir}")

if __name__ == "__main__":
    remove_confidence_from_yolo_segmentation_labels(LABEL_DIR, OUT_DIR)
