import os
import shutil
import re
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm


_NUMBER_PATTERN = re.compile(r"(\d+)")


def _numerical_sort(value: str):
    parts = _NUMBER_PATTERN.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_sorted(dir_path: str) -> List[str]:
    return sorted(os.listdir(dir_path), key=_numerical_sort)


def _pair_lists(images: List[str], sem_masks: List[str], inst_masks: List[str]) -> None:
    assert len(images) == len(sem_masks) == len(inst_masks), "Mismatched counts across images/sem_masks/inst_masks"


def _split(
    items_a: List[str],
    items_b: List[str],
    items_c: List[str],
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    a_trainval, a_test, b_trainval, b_test, c_trainval, c_test = train_test_split(
        items_a, items_b, items_c, test_size=test_size, random_state=seed
    )
    a_train, a_val, b_train, b_val, c_train, c_val = train_test_split(
        a_trainval, b_trainval, c_trainval, test_size=val_size, random_state=seed
    )
    return (a_train, a_val, a_test), (b_train, b_val, b_test), (c_train, c_val, c_test)


def _copy_split(
    src_img_dir: str,
    src_sem_dir: str,
    src_inst_dir: str,
    split_lists: Tuple[List[str], List[str], List[str]],
    dst_root: str,
    split_name: str,
) -> None:
    img_list, sem_list, inst_list = split_lists

    dst_img = os.path.join(dst_root, split_name, "images")
    dst_sem = os.path.join(dst_root, split_name, "sem_masks")
    dst_inst = os.path.join(dst_root, split_name, "inst_masks")
    _ensure_dir(dst_img)
    _ensure_dir(dst_sem)
    _ensure_dir(dst_inst)

    for img, sem, inst in tqdm(
        list(zip(img_list, sem_list, inst_list)),
        total=len(img_list),
        desc=f"Copying {split_name}",
    ):
        shutil.copy2(os.path.join(src_img_dir, img), dst_img)
        shutil.copy2(os.path.join(src_sem_dir, sem), dst_sem)
        shutil.copy2(os.path.join(src_inst_dir, inst), dst_inst)


def _save_split_lists(dst_root: str, name: str, items: List[str]) -> None:
    _ensure_dir(dst_root)
    with open(os.path.join(dst_root, f"{name}.txt"), "w") as f:
        for item in items:
            f.write(f"{item}\n")


def choose_top_tissues(processed_root: str, top_k: int = 5) -> List[str]:
    counts = []
    for tissue in os.listdir(processed_root):
        img_dir = os.path.join(processed_root, tissue, "images")
        if not os.path.isdir(img_dir):
            continue
        count = sum(1 for _ in os.scandir(img_dir))
        counts.append((count, tissue))
    counts.sort(reverse=True)
    return [t for _, t in counts[:top_k]]


def split_by_tissue(
    processed_root: str,
    dataset_root: str,
    tissues: List[str],
    test_split: float = 0.20,
    val_split: float = 0.10,
    seed: int = 42,
) -> None:
    _ensure_dir(dataset_root)
    for tissue in tissues:
        src_img_dir = os.path.join(processed_root, tissue, "images")
        src_sem_dir = os.path.join(processed_root, tissue, "sem_masks")
        src_inst_dir = os.path.join(processed_root, tissue, "inst_masks")
        if not (os.path.isdir(src_img_dir) and os.path.isdir(src_sem_dir) and os.path.isdir(src_inst_dir)):
            print(f"Skipping {tissue}: missing required directories")
            continue

        img_list = _list_sorted(src_img_dir)
        sem_list = _list_sorted(src_sem_dir)
        inst_list = _list_sorted(src_inst_dir)
        _pair_lists(img_list, sem_list, inst_list)

        (img_train, img_val, img_test), (sem_train, sem_val, sem_test), (inst_train, inst_val, inst_test) = _split(
            img_list, sem_list, inst_list, test_split, val_split, seed
        )

        dst_tissue_root = os.path.join(dataset_root, tissue)
        _copy_split(src_img_dir, src_sem_dir, src_inst_dir, (img_train, sem_train, inst_train), dst_tissue_root, "train")
        _copy_split(src_img_dir, src_sem_dir, src_inst_dir, (img_val, sem_val, inst_val), dst_tissue_root, "val")
        _copy_split(src_img_dir, src_sem_dir, src_inst_dir, (img_test, sem_test, inst_test), dst_tissue_root, "test")

        _save_split_lists(dst_tissue_root, "train", img_train)
        _save_split_lists(dst_tissue_root, "val", img_val)
        _save_split_lists(dst_tissue_root, "test", img_test)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_root = os.path.join(project_root, "output")
    dataset_root = os.path.join(project_root, "dataset_tissues")
    tissues = choose_top_tissues(processed_root, top_k=5)
    print("Selected tissues:", tissues)
    split_by_tissue(processed_root, dataset_root, tissues)


if __name__ == "__main__":
    main()


