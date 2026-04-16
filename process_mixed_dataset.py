import os
import glob
import numpy as np
import json
import gzip
import random
from split import split_document
from mask import mask_blocks, fill_masks, simhash_similarity, estimate_ppl, calculate_direct_ppl

def load_txt_books(data_dir, max_files=50):
    """
    随机加载指定目录下的txt文件
    data_dir: 数据目录路径
    max_files: 最多加载的文件数量，默认50
    """
    books = []
    book_paths = []
    
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 随机抽取max_files个文件
    if len(txt_files) > max_files:
        selected_files = random.sample(txt_files, max_files)
    else:
        selected_files = txt_files
    
    print(f"随机抽取 {len(selected_files)} 个文件进行处理")
    
    for file_path in selected_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            books.append(text)
            book_paths.append(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return books, book_paths

def process_book(book_text, max_tokens=512, filter_method=None, top_ratio=0.2):
    """
    对一本书进行分割，可选的关键段落筛选，并计算每段的相似度和困惑度
    
    book_text: 书籍文本
    max_tokens: 每段最大token数
    filter_method: 筛选方法，'uniqueness'（方法一）或 'density'（方法二），None表示不筛选
    top_ratio: 选择前多少比例的段落，默认0.2
    
    returns: (平均汉明距离, 平均困惑度, 筛选耗时, 原始段落数, 筛选后段落数)
    """
    chunks = split_document(book_text, max_tokens)
    print(f"分割为 {len(chunks)} 个片段")
    original_chunk_count = len(chunks)
    
    filter_time = 0
    
    # 如果指定了筛选方法，则进行关键段落筛选
    if filter_method:
        from key_segment_filter import filter_segments_by_method
        print(f"使用{filter_method}方法进行关键段落筛选...")
        chunks, _, filter_time = filter_segments_by_method(chunks, method=filter_method, top_ratio=top_ratio)
        print(f"筛选后剩余 {len(chunks)} 个关键片段")
    
    Total_hamming_distance = 0
    Total_ppl = 0
    processed_count = 0
    
    for i, chunk in enumerate(chunks):
        print(f"处理第 {i+1}/{len(chunks)} 段")
        tokens = chunk.split()
        if len(tokens) < 10:  # 跳过太短的片段
            continue
            
        masked_tokens, mask_spans = mask_blocks(tokens, mask_ratio=0.15)
        augmented = " ".join(fill_masks(masked_tokens, mask_spans, 512))
        hamming_distance = simhash_similarity(chunk, augmented)
        ppl = calculate_direct_ppl(augmented)
        
        Total_hamming_distance += hamming_distance
        Total_ppl += ppl
        processed_count += 1
    
    if processed_count == 0:
        return 0, 0, filter_time, original_chunk_count, len(chunks)
    
    Avg_hamming_distance = Total_hamming_distance / processed_count
    Avg_ppl = Total_ppl / processed_count
    return Avg_hamming_distance, Avg_ppl, filter_time, original_chunk_count, len(chunks)

def process_with_method(pg19_books, pg19_paths, nonmember_books, nonmember_paths, 
                        filter_method=None, method_name="no_filter"):
    """
    使用指定的筛选方法处理数据集
    
    filter_method: None, 'uniqueness', 或 'density'
    method_name: 方法名称，用于生成输出文件名
    """
    print(f"\n{'='*60}")
    print(f"使用方法: {method_name}")
    print(f"{'='*60}")
    
    all_features = []
    all_labels = []
    total_filter_time = 0
    total_original_chunks = 0
    total_filtered_chunks = 0
    
    # 处理PG19书籍（成员数据）
    print(f"\n=== 处理PG19书籍（{len(pg19_books)}本）===")
    for i, book in enumerate(pg19_books):
        print(f"Processing PG19 book {i+1}/{len(pg19_books)}: {os.path.basename(pg19_paths[i])}")
        try:
            hamming_distance, ppl, filter_time, orig_chunks, filt_chunks = process_book(
                book, filter_method=filter_method
            )
            all_features.extend([hamming_distance, ppl])
            all_labels.append(1)  # PG19标记为成员
            total_filter_time += filter_time
            total_original_chunks += orig_chunks
            total_filtered_chunks += filt_chunks
        except Exception as e:
            print(f"Error processing PG19 book {i+1}: {e}")
            all_features.extend([0, 0])  # 出错时用默认值
            all_labels.append(1)
    
    # 处理非成员书籍
    print(f"\n=== 处理非成员书籍（{len(nonmember_books)}本）===")
    for i, book in enumerate(nonmember_books):
        print(f"Processing non-member book {i+1}/{len(nonmember_books)}: {os.path.basename(nonmember_paths[i])}")
        try:
            hamming_distance, ppl, filter_time, orig_chunks, filt_chunks = process_book(
                book, filter_method=filter_method
            )
            all_features.extend([hamming_distance, ppl])
            all_labels.append(0)  # 非成员标记
            total_filter_time += filter_time
            total_original_chunks += orig_chunks
            total_filtered_chunks += filt_chunks
        except Exception as e:
            print(f"Error processing non-member book {i+1}: {e}")
            all_features.extend([0, 0])  # 出错时用默认值
            all_labels.append(0)
    
    all_features = np.array(all_features)
    
    # 生成输出文件名
    feature_file = f"gutenberg_features_{method_name}.txt"
    label_file = f"gutenberg_labels_{method_name}.txt"
    efficiency_file = f"efficiency_{method_name}.txt"
    
    # 保存特征向量到txt文件
    with open(feature_file, "w") as f:
        for i in range(0, len(all_features), 2):
            f.write(f"{all_features[i]},{all_features[i+1]}\n")
    
    # 保存标签
    with open(label_file, "w") as f:
        for label in all_labels:
            f.write(f"{label}\n")
    
    # 保存效率信息
    avg_filter_time = total_filter_time / len(all_labels) if len(all_labels) > 0 else 0
    reduction_ratio = (1 - total_filtered_chunks / total_original_chunks) * 100 if total_original_chunks > 0 else 0
    
    with open(efficiency_file, "w") as f:
        f.write(f"筛选方法: {method_name}\n")
        f.write(f"总筛选耗时: {total_filter_time:.2f}秒\n")
        f.write(f"平均每本书筛选耗时: {avg_filter_time:.2f}秒\n")
        f.write(f"原始段落总数: {total_original_chunks}\n")
        f.write(f"筛选后段落总数: {total_filtered_chunks}\n")
        f.write(f"段落减少比例: {reduction_ratio:.2f}%\n")
    
    print(f"\n=== {method_name} 处理完成 ===")
    print(f"总书籍数: {len(all_features) // 2}")
    print(f"成员数据: {sum(all_labels)} 本")
    print(f"非成员数据: {len(all_labels) - sum(all_labels)} 本")
    print(f"总筛选耗时: {total_filter_time:.2f}秒")
    print(f"平均每本书筛选耗时: {avg_filter_time:.2f}秒")
    print(f"段落减少比例: {reduction_ratio:.2f}%")
    print(f"特征文件已保存: {feature_file}")
    print(f"标签文件已保存: {label_file}")
    print(f"效率信息已保存: {efficiency_file}")
    
    return feature_file, label_file, efficiency_file

def main():
    # 加载PG19数据集（成员数据，label=1）
    print("=== 加载PG19数据集（成员数据）===")
    pg19_books, pg19_paths = load_txt_books("/home/junyi/MASK/data/raw_gutenberg/pg19_downloads", max_files=50)
    
    # 加载非成员数据集（label=0）
    print("=== 加载非成员数据集 ===")
    nonmember_books, nonmember_paths = load_txt_books("/home/junyi/MASK/data/raw_gutenberg/data", max_files=50)
    
    import time
    start_time = time.time()
    
    # 方法对比：三种情况
    results = {}
    
    # 1. 无筛选（baseline）
    '''print("\n" + "="*60)
    print("实验1: 无筛选（Baseline）")
    print("="*60)
    method_start = time.time()
    feature_file, label_file, eff_file = process_with_method(
        pg19_books, pg19_paths, nonmember_books, nonmember_paths,
        filter_method=None, method_name="no_filter"
    )
    results["no_filter"] = {
        "feature_file": feature_file,
        "label_file": label_file,
        "efficiency_file": eff_file,
        "total_time": time.time() - method_start
    }'''
    
    # 2. 方法一：基于语句独特性特征
    print("\n" + "="*60)
    print("实验2: 方法一 - 基于语句独特性特征")
    print("="*60)
    method_start = time.time()
    feature_file, label_file, eff_file = process_with_method(
        pg19_books, pg19_paths, nonmember_books, nonmember_paths,
        filter_method='uniqueness', method_name="uniqueness"
    )
    results["uniqueness"] = {
        "feature_file": feature_file,
        "label_file": label_file,
        "efficiency_file": eff_file,
        "total_time": time.time() - method_start
    }
    
    # 3. 方法二：基于专有名词占比
    print("\n" + "="*60)
    print("实验3: 方法二 - 基于专有名词占比")
    print("="*60)
    method_start = time.time()
    feature_file, label_file, eff_file = process_with_method(
        pg19_books, pg19_paths, nonmember_books, nonmember_paths,
        filter_method='density', method_name="density"
    )
    results["density"] = {
        "feature_file": feature_file,
        "label_file": label_file,
        "efficiency_file": eff_file,
        "total_time": time.time() - method_start
    }
    
    total_time = time.time() - start_time
    
    # 生成对比报告
    print("\n" + "="*60)
    print("实验对比总结")
    print("="*60)
    print(f"总耗时: {total_time:.2f}秒")
    print("\n各方法耗时对比:")
    for method, res in results.items():
        print(f"  {method}: {res['total_time']:.2f}秒")
    
    # 保存对比报告
    with open("comparison_report.txt", "w") as f:
        f.write("关键段落筛选方法对比报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"总实验耗时: {total_time:.2f}秒\n\n")
        f.write("各方法详细信息:\n")
        for method, res in results.items():
            f.write(f"\n方法: {method}\n")
            f.write(f"  特征文件: {res['feature_file']}\n")
            f.write(f"  标签文件: {res['label_file']}\n")
            f.write(f"  效率文件: {res['efficiency_file']}\n")
            f.write(f"  总耗时: {res['total_time']:.2f}秒\n")
    
    print("\n对比报告已保存: comparison_report.txt")
    print("\n接下来可以使用 membership_detection.py 评估每种方法的AUC性能")

if __name__ == "__main__":
    main()
