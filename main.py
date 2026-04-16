import argparse
import glob
import os
import random
from typing import Callable

from key_segment_filter import filter_segments_by_method
from mask import (
    calculate_direct_ppl,
    calculate_lowercase_score,
    calculate_min_k_prob_score,
    calculate_prefix_suffix_ppl,
    calculate_zlib_score,
    fill_masks,
    mask_blocks,
    simhash_similarity,
)
from split import split_document


def load_txt_books(data_dir, max_files=1000, min_tokens=5000, seed=42):
    books = []
    paths = []
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if len(txt_files) == 0:
        print(f"警告：目录为空或不存在 txt 文件: {data_dir}")
        return books, paths

    rng = random.Random(seed)
    if len(txt_files) > max_files:
        txt_files = rng.sample(txt_files, max_files)

    for p in txt_files:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read()
            if len(t.split()) < min_tokens:
                continue
            books.append(t)
            paths.append(p)
        except Exception as e:
            print(f"加载失败 {p}: {e}")
    return books, paths


def process_dotmi_book(book_text, max_tokens=512, filter_method=None, top_ratio=0.2, mask_ratio=0.15):
    chunks = split_document(book_text, max_tokens)
    if len(chunks) == 0:
        return 0.0, 0.0

    if filter_method:
        chunks, _, _ = filter_segments_by_method(chunks, method=filter_method, top_ratio=top_ratio)

    total_hamming = 0.0
    total_ppl = 0.0
    valid = 0
    for chunk in chunks:
        tokens = chunk.split()
        if len(tokens) < 10:
            continue
        masked_tokens, mask_spans = mask_blocks(tokens, mask_ratio=mask_ratio)
        generated = " ".join(fill_masks(masked_tokens, mask_spans, max_tokens))
        total_hamming += simhash_similarity(chunk, generated)
        total_ppl += calculate_direct_ppl(generated)
        valid += 1

    if valid == 0:
        return 0.0, 0.0
    return total_hamming / valid, total_ppl / valid


def process_baseline_book(book_text, baseline_name):
    # 一维 baseline 分数；统一约定：分数越小越偏成员
    if baseline_name == "ppl":
        return calculate_direct_ppl(book_text)
    if baseline_name == "min_k":
        return calculate_min_k_prob_score(book_text, k_ratio=0.2)
    if baseline_name == "zlib":
        return calculate_zlib_score(book_text)
    if baseline_name == "lowercase":
        return calculate_lowercase_score(book_text)
    if baseline_name == "prefix_suffix_ppl":
        return calculate_prefix_suffix_ppl(book_text, prefix_tokens=256)
    raise ValueError(f"未知 baseline: {baseline_name}")


def save_features_and_labels(features, labels, feature_file, label_file):
    with open(feature_file, "w") as f:
        for row in features:
            f.write(",".join(str(x) for x in row) + "\n")
    with open(label_file, "w") as f:
        for y in labels:
            f.write(f"{y}\n")


def run_one_experiment(
    dataset_name,
    member_dir,
    nonmember_dir,
    mode,
    max_files=200,
    min_tokens=5000,
    seed=42,
    max_tokens=512,
    top_ratio=0.2,
    mask_ratio=0.15,
):
    print(f"\n{'=' * 72}")
    print(f"数据集: {dataset_name} | 模式: {mode}")
    print(f"{'=' * 72}")

    member_books, member_paths = load_txt_books(member_dir, max_files=max_files, min_tokens=min_tokens, seed=seed)
    nonmember_books, nonmember_paths = load_txt_books(
        nonmember_dir, max_files=max_files, min_tokens=min_tokens, seed=seed + 1
    )

    print(f"成员文档: {len(member_books)} | 非成员文档: {len(nonmember_books)}")
    if len(member_books) == 0 or len(nonmember_books) == 0:
        print("数据不足，跳过该实验。")
        return None, None

    features = []
    labels = []

    if mode in {"dotmi_no_filter", "dotmi_uniqueness", "dotmi_density"}:
        filter_method = None
        if mode == "dotmi_uniqueness":
            filter_method = "uniqueness"
        elif mode == "dotmi_density":
            filter_method = "density"

        for i, text in enumerate(member_books):
            print(f"[member {i+1}/{len(member_books)}] {os.path.basename(member_paths[i])}")
            d, p = process_dotmi_book(
                text,
                max_tokens=max_tokens,
                filter_method=filter_method,
                top_ratio=top_ratio,
                mask_ratio=mask_ratio,
            )
            features.append([d, p])
            labels.append(1)

        for i, text in enumerate(nonmember_books):
            print(f"[non-member {i+1}/{len(nonmember_books)}] {os.path.basename(nonmember_paths[i])}")
            d, p = process_dotmi_book(
                text,
                max_tokens=max_tokens,
                filter_method=filter_method,
                top_ratio=top_ratio,
                mask_ratio=mask_ratio,
            )
            features.append([d, p])
            labels.append(0)
    else:
        # baseline: 输出单列分数
        for i, text in enumerate(member_books):
            print(f"[member {i+1}/{len(member_books)}] {os.path.basename(member_paths[i])}")
            s = process_baseline_book(text, mode)
            features.append([s])
            labels.append(1)
        for i, text in enumerate(nonmember_books):
            print(f"[non-member {i+1}/{len(nonmember_books)}] {os.path.basename(nonmember_paths[i])}")
            s = process_baseline_book(text, mode)
            features.append([s])
            labels.append(0)

    feature_file = f"{dataset_name}_features_{mode}.txt"
    label_file = f"{dataset_name}_labels_{mode}.txt"
    save_features_and_labels(features, labels, feature_file, label_file)
    print(f"已保存: {feature_file}, {label_file}")
    return feature_file, label_file


def evaluate_and_report(feature_file, label_file):
    from membership_detection import evaluate_membership_detection

    if feature_file is None or label_file is None:
        return None
    return evaluate_membership_detection(feature_file, label_file)


def main():
    parser = argparse.ArgumentParser(description="DoTMI + baseline 统一运行脚本（保留原文件架构）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--min_tokens", type=int, default=5000)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_ratio", type=float, default=0.2)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "dotmi_uniqueness",
            "dotmi_density",
            "ppl",
            "min_k",
            "zlib",
            "lowercase",
            "prefix_suffix_ppl",
        ],
        help="可选: dotmi_no_filter dotmi_uniqueness dotmi_density ppl min_k zlib lowercase prefix_suffix_ppl",
    )
    parser.add_argument("--run_gutenberg", action="store_true")
    parser.add_argument("--run_arxiv", action="store_true")
    parser.add_argument("--g_member_dir", default="/home/junyi/MASK/data/raw_gutenberg/pg19_downloads")
    parser.add_argument("--g_nonmember_dir", default="/home/junyi/MASK/data/raw_gutenberg/data")
    parser.add_argument("--a_member_dir", default="/home/junyi/MASK/data/raw_arxiv/member")
    parser.add_argument("--a_nonmember_dir", default="/home/junyi/MASK/data/raw_arxiv/non_member")
    args = parser.parse_args()

    run_g = args.run_gutenberg or (not args.run_gutenberg and not args.run_arxiv)
    run_a = args.run_arxiv or (not args.run_gutenberg and not args.run_arxiv)

    all_results = {}
    for dataset_name, member_dir, nonmember_dir, enabled in [
        ("gutenberg", args.g_member_dir, args.g_nonmember_dir, run_g),
        ("arxiv", args.a_member_dir, args.a_nonmember_dir, run_a),
    ]:
        if not enabled:
            continue
        all_results[dataset_name] = {}
        for mode in args.modes:
            ff, lf = run_one_experiment(
                dataset_name=dataset_name,
                member_dir=member_dir,
                nonmember_dir=nonmember_dir,
                mode=mode,
                max_files=args.max_files,
                min_tokens=args.min_tokens,
                seed=args.seed,
                max_tokens=args.max_tokens,
                top_ratio=args.top_ratio,
                mask_ratio=args.mask_ratio,
            )
            metrics = evaluate_and_report(ff, lf)
            all_results[dataset_name][mode] = {
                "feature_file": ff,
                "label_file": lf,
                "metrics": metrics,
            }

    print("\n实验完成。结果摘要：")
    for ds, modes in all_results.items():
        print(f"\n[{ds}]")
        for m, info in modes.items():
            auc = None
            if info["metrics"] is not None:
                # evaluate_membership_detection 返回 (best_theta_or_none, auc, tpr@fpr)
                auc = info["metrics"][1]
            print(f"  {m:<20} AUC={auc}")


if __name__ == "__main__":
    main()
