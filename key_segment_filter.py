"""
关键段落筛选模块
实现两种方法：
1. 方法一：基于语句独特性特征（BERT嵌入向量 + 余弦相似度）
2. 方法二：基于专有名词占比（信息密度指标）
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch
import nltk
from nltk import pos_tag, word_tokenize
import re
import time
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# 加载BERT模型（用于方法一）
print("正在加载BERT模型...")
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name)
bert_model.eval()

# 常见人名、地名列表（用于方法二，过滤常见词）
COMMON_NAMES = {
    'john', 'mary', 'james', 'robert', 'william', 'richard', 'thomas', 'charles',
    'david', 'michael', 'joseph', 'daniel', 'christopher', 'andrew', 'joseph',
    'mark', 'donald', 'steven', 'paul', 'anthony', 'kenneth', 'joshua', 'kevin',
    'brian', 'george', 'timothy', 'ronald', 'edward', 'jason', 'matthew', 'jeffrey',
    'ryan', 'jacob', 'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry',
    'justin', 'scott', 'brandon', 'benjamin', 'frank', 'gregory', 'raymond',
    'alexander', 'patrick', 'jack', 'dennis', 'jerry', 'tyler', 'aaron', 'jose',
    'henry', 'adam', 'douglas', 'nathan', 'peter', 'zachary', 'kyle', 'noah',
    'ethan', 'jeremy', 'walter', 'christian', 'keith', 'roger', 'terry', 'austin',
    'sean', 'gerald', 'carl', 'arthur', 'juan', 'lawrence', 'dylan', 'jesse',
    'jordan', 'bryan', 'billy', 'joe', 'bruce', 'gabriel', 'logan', 'albert',
    'ralph', 'randy', 'eugene', 'wayne', 'roy', 'louis', 'philip', 'bobby',
    'johnny', 'carlos', 'willie', 'harold', 'alan', 'angel', 'howard', 'samuel',
    'eddie', 'jack', 'williams', 'victor', 'lawrence', 'nicholas', 'roy', 'clarence',
    'samuel', 'russell', 'bobby', 'mason', 'philip', 'christopher', 'edward'
}

COMMON_PLACES = {
    'london', 'paris', 'new york', 'berlin', 'rome', 'madrid', 'amsterdam',
    'vienna', 'moscow', 'tokyo', 'beijing', 'shanghai', 'sydney', 'melbourne',
    'toronto', 'vancouver', 'chicago', 'los angeles', 'san francisco', 'boston',
    'washington', 'philadelphia', 'houston', 'miami', 'seattle', 'detroit',
    'atlanta', 'dallas', 'denver', 'phoenix', 'las vegas', 'portland', 'nashville',
    'england', 'france', 'germany', 'italy', 'spain', 'russia', 'china', 'japan',
    'canada', 'australia', 'america', 'united states', 'usa', 'uk', 'britain'
}

def get_bert_embedding(text, max_length=512):
    """
    使用BERT获取文本的嵌入向量
    """
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, 
                           max_length=max_length, padding=True)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # 使用[CLS] token的嵌入，或者对所有token嵌入取平均
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    
    return embeddings.numpy()

def get_wikitext_corpus_center(max_samples=1000):
    """
    从 WikiText 样本计算通用语料库嵌入中心（与方法一内部逻辑一致）。
    便于在多篇文档上复用同一中心，使独特性分数可比。
    """
    print("  加载WikiText数据集以计算语料库中心...")
    wiki_samples = []

    if DATASETS_AVAILABLE:
        try:
            try:
                print("    尝试加载WikiText-103...")
                dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            except Exception:
                print("    WikiText-103不可用，尝试加载WikiText-2...")
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

            print(f"    从数据集中提取样本（共{len(dataset)}条）...")
            sample_count = 0
            for item in dataset:
                text = item.get('text', '').strip()
                if text and len(text) > 50 and not text.startswith('='):
                    wiki_samples.append(text)
                    sample_count += 1
                    if sample_count >= max_samples:
                        break
            print(f"    成功提取 {len(wiki_samples)} 个WikiText样本")
        except Exception as e:
            print(f"    加载WikiText数据集失败: {e}")
            print("    回退到使用默认样本...")
            wiki_samples = [
                "The quick brown fox jumps over the lazy dog.",
                "Natural language processing is a field of artificial intelligence.",
                "Machine learning algorithms learn from data.",
                "The history of the world is written in books.",
                "Science and technology have transformed society.",
                "Literature and art reflect human culture.",
                "Mathematics is the language of the universe.",
                "Philosophy explores fundamental questions about existence."
            ] * 10
    else:
        print("    datasets库不可用，使用默认样本...")
        wiki_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is a field of artificial intelligence.",
            "Machine learning algorithms learn from data.",
            "The history of the world is written in books.",
            "Science and technology have transformed society.",
            "Literature and art reflect human culture.",
            "Mathematics is the language of the universe.",
            "Philosophy explores fundamental questions about existence."
        ] * 10

    print("  计算语料库嵌入中心...")
    wiki_embeddings = []
    for i, sample in enumerate(wiki_samples):
        if i % 100 == 0:
            print(f"    处理样本 {i+1}/{len(wiki_samples)}")
        try:
            emb = get_bert_embedding(sample)
            wiki_embeddings.append(emb)
        except Exception as e:
            print(f"    处理样本 {i+1} 时出错: {e}，跳过")
            continue

    if len(wiki_embeddings) == 0:
        raise ValueError("无法计算语料库嵌入中心：没有有效的嵌入向量")
    corpus_center = np.mean(wiki_embeddings, axis=0)
    print(f"  成功计算语料库嵌入中心（基于{len(wiki_embeddings)}个样本）")
    return corpus_center

def compute_uniqueness_score_method1(segments, corpus_center=None):
    """
    方法一：基于语句独特性特征
    计算每个段落与通用语料库嵌入中心的余弦相似度
    相似度越低，独特性越高，U_i = 1 - sim_avg(S_i)
    
    segments: 段落列表
    corpus_center: 通用语料库的嵌入中心向量（如果为None，则使用WikiText样本计算）
    """
    print("方法一：计算语句独特性特征...")
    start_time = time.time()

    if corpus_center is None:
        corpus_center = get_wikitext_corpus_center()

    # 计算每个段落的嵌入和独特性分数
    uniqueness_scores = []
    segment_embeddings = []
    
    for i, segment in enumerate(segments):
        if i % 10 == 0:
            print(f"  处理段落 {i+1}/{len(segments)}")
        
        # 获取段落嵌入
        seg_emb = get_bert_embedding(segment)
        segment_embeddings.append(seg_emb)
        
        # 计算与语料库中心的余弦相似度
        similarity = cosine_similarity([seg_emb], [corpus_center])[0][0]
        
        # 独特性分数 = 1 - 相似度
        uniqueness_score = 1 - similarity
        uniqueness_scores.append(uniqueness_score)
    
    elapsed_time = time.time() - start_time
    print(f"方法一计算完成，耗时: {elapsed_time:.2f}秒")
    
    return np.array(uniqueness_scores), elapsed_time

def extract_proper_nouns(text):
    """
    提取文本中的专有名词
    使用POS tagging识别专有名词，并过滤常见词
    """
    # 分词和词性标注
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    proper_nouns = []
    current_phrase = []
    
    for token, pos in pos_tags:
        # NNP (单数专有名词) 或 NNS (复数专有名词，但这里我们主要关注NNP)
        if pos == 'NNP':
            current_phrase.append(token.lower())
        else:
            if current_phrase:
                phrase = ' '.join(current_phrase)
                # 过滤常见人名地名
                if phrase not in COMMON_NAMES and phrase not in COMMON_PLACES:
                    # 过滤单个常见词
                    if len(current_phrase) > 1 or current_phrase[0] not in COMMON_NAMES:
                        proper_nouns.append(phrase)
                current_phrase = []
            elif pos.startswith('NN') and token[0].isupper():
                # 处理可能遗漏的专有名词（首字母大写）
                token_lower = token.lower()
                if token_lower not in COMMON_NAMES and token_lower not in COMMON_PLACES:
                    proper_nouns.append(token_lower)
    
    # 处理末尾的专有名词
    if current_phrase:
        phrase = ' '.join(current_phrase)
        if phrase not in COMMON_NAMES and phrase not in COMMON_PLACES:
            proper_nouns.append(phrase)
    
    return proper_nouns

def compute_information_density_method2(segments):
    """
    方法二：基于专有名词占比计算信息密度
    计算每个段落中专有名词的占比作为信息密度指标 D_i
    
    segments: 段落列表
    """
    print("方法二：计算专有名词占比（信息密度）...")
    start_time = time.time()
    
    information_densities = []
    
    for i, segment in enumerate(segments):
        if i % 10 == 0:
            print(f"  处理段落 {i+1}/{len(segments)}")
        
        # 提取专有名词
        proper_nouns = extract_proper_nouns(segment)
        
        # 计算总词数
        tokens = word_tokenize(segment)
        total_words = len([t for t in tokens if t.isalnum()])  # 只统计字母数字词
        
        # 计算信息密度 = 专有名词数量 / 总词数
        if total_words > 0:
            density = len(proper_nouns) / total_words
        else:
            density = 0.0
        
        information_densities.append(density)
    
    elapsed_time = time.time() - start_time
    print(f"方法二计算完成，耗时: {elapsed_time:.2f}秒")
    
    return np.array(information_densities), elapsed_time

def filter_key_segments(segments, scores, top_ratio=0.2):
    """
    根据分数筛选出前top_ratio比例的关键段落
    
    segments: 段落列表
    scores: 分数数组（独特性分数或信息密度）
    top_ratio: 选择前多少比例的段落，默认0.2（20%）
    
    returns: 筛选后的段落列表和索引列表
    """
    num_select = max(1, int(len(segments) * top_ratio))
    
    # 获取分数最高的索引
    top_indices = np.argsort(scores)[-num_select:][::-1]
    
    # 筛选段落
    key_segments = [segments[i] for i in top_indices]
    
    return key_segments, top_indices.tolist()

def filter_segments_by_method(segments, method='uniqueness', top_ratio=0.2, corpus_center=None):
    """
    根据指定方法筛选关键段落
    
    segments: 段落列表
    method: 'uniqueness' (方法一) 或 'density' (方法二)
    top_ratio: 选择前多少比例的段落，默认0.2
    corpus_center: 方法一的语料库嵌入中心（可选）
    
    returns: (筛选后的段落列表, 使用的分数数组, 计算耗时)
    """
    if method == 'uniqueness':
        scores, elapsed_time = compute_uniqueness_score_method1(segments, corpus_center)
    elif method == 'density':
        scores, elapsed_time = compute_information_density_method2(segments)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'uniqueness' or 'density'")
    
    key_segments, indices = filter_key_segments(segments, scores, top_ratio)
    
    print(f"从 {len(segments)} 个段落中筛选出 {len(key_segments)} 个关键段落 ({top_ratio*100:.1f}%)")
    
    return key_segments, scores, elapsed_time