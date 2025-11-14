# import os
# import json

# def merge_json_files(base_dir, subdirs, output_dir):
#     # 첫 번째 서브 디렉토리에서 모든 JSON 파일 목록을 가져옴
#     first_subdir_path = os.path.join(base_dir, subdirs[0])
#     all_files = [f for f in os.listdir(first_subdir_path) if f.endswith('.json')]

#     for filename in all_files:
#         merged_shapes = []
#         version = None
#         flags = None

#         # 각 서브 디렉토리별로 해당 JSON 파일을 읽어들임
#         for subdir in subdirs:
#             file_path = os.path.join(base_dir, subdir, filename)
#             if os.path.exists(file_path):  # 파일이 존재하는 경우에만 읽음
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)
#                     if "shapes" in data:
#                         merged_shapes.extend(data["shapes"])
#                     if not version and "version" in data:
#                         version = data["version"]
#                     if not flags and "flags" in data:
#                         flags = data["flags"]

#         # 모든 데이터를 하나의 파일에 저장
#         merged_data = {
#             "version": version,
#             "flags": flags,
#             "shapes": merged_shapes
#         }
#         output_path = os.path.join(output_dir, filename)
#         with open(output_path, 'w') as f:
#             json.dump(merged_data, f, ensure_ascii=False, indent=2)  # indent 값을 2로 변경

# if __name__ == "__main__":
#     subdirs = ['worker', 'hardhat', 'harness', 'strap', 'hook']

#     # test1부터 test4까지 반복
#     for i in range(1, 5):
#         base_dir = f'C:/Users/test1/Desktop/New_daaset/New_dataset_final/sam_output/json/test{i}'
#         output_dir = f'C:/Users/test1/Desktop/New_daaset/New_dataset_final/sam_output/json/conbine_json/test{i}'
        
#         # 해당 디렉토리가 존재하지 않으면 생성
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         merge_json_files(base_dir, subdirs, output_dir)
#         print(f"{output_dir}에 모든 파일들이 합쳐졌습니다.")



from pathlib import Path
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image

# ===== 경로 설정 =====
BASE    = r"C:\Users\KUMHOENC\OneDrive - smartkumho\바탕 화면\개인\1. 연구실\0. 저널\2. 리비전\2. 답변서\2차 코멘트\실험"
DS_ROOT = rf"{BASE}\dataset\ADD_EXPERIMENTS_DATASET"

TARGETS = [
    r"target5(Target5,7)",
    r"target6(Target6)",
    r"target7(Weak39~40)",
    r"target10(Weak41)",
]
SUBSETS = ["train", "val"]
CLASSES = ["worker", "hardhat", "strap", "hook"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def collect_by_filename(each_data_root: Path) -> Dict[str, List[Path]]:
    """each_data/{cls}/*.json을 파일명 기준으로 그룹핑."""
    bucket: Dict[str, List[Path]] = defaultdict(list)
    for cls in CLASSES:
        cls_dir = each_data_root / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.json"):
            bucket[p.name].append(p)
    return bucket

def try_pil_size(img_path: Path) -> Optional[Tuple[int, int]]:
    """PIL로 이미지 크기를 읽음 (w,h 반환). 실패 시 None."""
    try:
        with Image.open(img_path) as im:
            return im.size  # (w, h)
    except Exception:
        return None

def find_image_with_hint(image_dir: Path, stem: str, hint_names: List[str]) -> Optional[Tuple[int, int, str]]:
    """
    1) hint file name들 (json 내부 imagePath) 로 먼저 시도
    2) stem + 확장자 조합으로 탐색
    성공 시 (w,h,filename) 반환
    """
    # 1) 힌트 우선
    for hint in hint_names:
        if not hint or not isinstance(hint, str):
            continue
        cand = image_dir / Path(hint).name
        if cand.exists():
            size = try_pil_size(cand)
            if size:
                w, h = size
                return int(w), int(h), cand.name

    # 2) stem + 다중 확장자
    for ext in IMG_EXTS:
        cand = image_dir / f"{stem}{ext}"
        if cand.exists():
            size = try_pil_size(cand)
            if size:
                w, h = size
                return int(w), int(h), cand.name

    return None

def merge_jsons_for_image(json_paths: List[Path], image_dir: Path, stem: str) -> Tuple[dict, bool]:
    """
    같은 이미지(stem)의 클래스별 JSON들을 병합.
    - shapes는 모두 합침
    - version/flags는 첫 JSON에서 가져오되, 없어도 OK
    - imageWidth/Height/Path는 원본 이미지를 읽어 채움 (실패시 None)
    - imageData는 항상 None (용량 절약)
    반환: (merged_json, found_image_flag)
    """
    merged_shapes: List[dict] = []
    version = None
    flags = None
    hint_names: List[str] = []

    for jp in json_paths:
        try:
            data = load_json(jp)
        except Exception:
            continue
        if version is None:
            version = data.get("version")
        if flags is None:
            flags = data.get("flags", {})
        sh = data.get("shapes", [])
        if isinstance(sh, list):
            merged_shapes.extend(sh)
        # imagePath 힌트 수집
        ip = data.get("imagePath")
        if isinstance(ip, str):
            hint_names.append(ip)

    meta = find_image_with_hint(image_dir, stem, hint_names)
    if meta:
        w, h, fname = meta
        found = True
    else:
        w = h = None
        fname = None
        found = False

    merged = {
        "version": version,
        "flags": flags if isinstance(flags, dict) else {},
        "shapes": merged_shapes,
        "imagePath": fname,       # 찾으면 파일명, 못 찾으면 None
        "imageData": None,        # 항상 None
        "imageWidth": w,
        "imageHeight": h,
    }
    return merged, found

def run_one_target(tgt_dirname: str, counters: dict, missing_log: List[str]):
    base_dir = Path(DS_ROOT) / tgt_dirname / "labels_SAM_json"
    each_root = base_dir / "each_data"
    sum_root  = base_dir / "sum_data"

    for subset in SUBSETS:
        each_subset_root = each_root / subset
        out_subset_root  = sum_root / subset
        image_dir        = Path(DS_ROOT) / tgt_dirname / "images" / subset

        if not each_subset_root.exists():
            continue

        groups = collect_by_filename(each_subset_root)
        if not groups:
            continue

        for fname, paths in tqdm(groups.items(), desc=f"{tgt_dirname}/{subset}", unit="img", leave=True):
            stem = Path(fname).stem
            merged, found = merge_jsons_for_image(paths, image_dir=image_dir, stem=stem)
            if not found:
                missing_log.append(f"{tgt_dirname}/{subset}  |  {fname}  |  이미지 미검출")
            save_json(out_subset_root / fname, merged)
            counters["merged"] += 1

def main():
    counters = {"merged": 0}
    missing: List[str] = []

    for tgt in TARGETS:
        run_one_target(tgt, counters, missing)

    print(f"\n[요약] 병합 JSON 생성: {counters['merged']}개")
    if missing:
        print(f"[주의] 이미지 메타를 못 채운 항목: {len(missing)}개")
        # 필요하면 아래 주석 해제해서 상세 목록 출력
        # for line in missing:
        #     print("  -", line)

if __name__ == "__main__":
    main()
