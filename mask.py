import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from simhash import Simhash
import math
import zlib

# 加载Llama模型和分词器（只加载一次，避免重复加载）
tokenizer = AutoTokenizer.from_pretrained("/home/junyi/meta-llama/Llama-2-7b-hf", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("/home/junyi/meta-llama/Llama-2-7b-hf")
model.eval()


def mask_blocks(tokens, mask_ratio=0.15, min_block=20, max_block=50):
    """
    随机选取若干连续区间，总计约mask_ratio比例的token，返回mask后的tokens和区间列表
    """
    total = len(tokens)
    num_to_mask = int(total * mask_ratio)
    masked = tokens[:]
    mask_spans = []
    masked_count = 0
    used = set()
    while masked_count < num_to_mask:
        block_len = min(random.randint(min_block, max_block), num_to_mask - masked_count)
        start = random.randint(0, total - block_len)
        # 检查是否与已有区间重叠
        overlap = False
        for s, e in mask_spans:
            if not (start + block_len <= s or start >= e):
                overlap = True
                break
        if overlap:
            continue
        mask_spans.append((start, start+block_len))
        masked[start:start+block_len] = ["[MASK]"] * block_len
        masked_count += block_len
    mask_spans.sort()
    return masked, mask_spans


def fill_masks(masked_tokens, mask_spans, max_tokens=512):
    """
    依次用Llama补全每个[MASK]块，返回完整新tokens
    """
    tokens = masked_tokens[:]
    for start, end in mask_spans:
        # 构造prompt：前后各取一段上下文
        left = " ".join(tokens[max(0, start-30):start])
        right = " ".join(tokens[end:min(len(tokens), end+30)])
        prompt = f"Context:{left} [MASK] {right}\nPlease complete the [MASK] part: "
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=end-start,
                do_sample=True,
                top_p=0.95,
                temperature=0.8
            )
        filled = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只取补全内容（去掉prompt部分）
        if filled.startswith(prompt):
            filled = filled[len(prompt):]
        # 简单分词
        filled_tokens = filled.strip().split()[:end-start]
        tokens[start:end] = filled_tokens
    return tokens


def mask_and_fill(text, mask_ratio=0.15, max_tokens=512):
    tokens = text.split()
    masked_tokens, mask_spans = mask_blocks(tokens, mask_ratio)
    #print(masked_tokens)
    #print(mask_spans)
    filled_tokens = fill_masks(masked_tokens, mask_spans, max_tokens)
    return " ".join(filled_tokens)


def simhash_similarity(text1, text2):
    # 简单分词，可根据需要替换为更复杂分词
    tokens1 = text1.split()
    tokens2 = text2.split()
    h1 = Simhash(tokens1)
    h2 = Simhash(tokens2)
    # 汉明距离越小越相似
    return h1.distance(h2)


def estimate_ppl(original_text, generated_text, mask_spans, k=10):
    """
    对每个新生成的分词ti，将其mask后输入模型，用top-k生成，统计排名，计算近似困惑度PPL。
    original_text: 原始文本
    generated_text: 生成文本
    mask_spans: 被mask的区间列表
    k: top-k生成
    """
    tokens = generated_text.split()
    log_probs = []
    N = 0
    for start, end in mask_spans:
        for i in range(start, end):
            masked_tokens = tokens[:]
            masked_tokens[i] = "[MASK]"
            left = " ".join(masked_tokens[max(0, i-30):i])
            right = " ".join(masked_tokens[i+1:min(len(tokens), i+30)])
            prompt = f"Context:{left} [MASK] {right}\nPlease complete the [MASK] part: "
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    top_k=k,
                    temperature=0.8
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 取top-k生成的token
            gen_tokens = generated.strip().split()
            ti = tokens[i]
            # 统计ti在top-k中的排名
            try:
                rank = gen_tokens.index(ti) + 1
            except ValueError:
                rank = k + 1  # 不在top-k中
            pi = 1.0 / rank
            log_probs.append(math.log(pi))
            N += 1
    if N == 0:
        return float('inf')
    ppl = math.exp(-sum(log_probs) / N)
    return ppl


def calculate_direct_ppl(generated_text):
    """
    直接计算生成文本在目标模型（LLM）上的困惑度
    使用标准的语言模型困惑度计算方法
    """
    # 对生成文本进行tokenization
    inputs = tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        # 获取模型的输出logits
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 计算每个token的负对数似然
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 计算平均负对数似然
        avg_neg_log_likelihood = loss.mean().item()
        
        # 困惑度 = exp(平均负对数似然)
        ppl = math.exp(avg_neg_log_likelihood)
    
    return ppl


def _token_level_nll(text, max_length=512):
    """
    计算每个位置 token 的负对数似然（NLL）列表。
    用于 Min-K% Prob 等 baseline。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return token_nll.detach().cpu().numpy().tolist()


def calculate_min_k_prob_score(text, k_ratio=0.2, max_length=512):
    """
    Min-K% Prob baseline（值越小通常表示越可能是成员）。
    返回最小 k% token 的平均 NLL。
    """
    nlls = _token_level_nll(text, max_length=max_length)
    if len(nlls) == 0:
        return float("inf")
    nlls = sorted(nlls)
    k = max(1, int(len(nlls) * k_ratio))
    return float(sum(nlls[:k]) / k)


def calculate_zlib_score(text, max_length=512):
    """
    zlib baseline：PPL / 压缩比（近似实现）。
    值越大通常越偏非成员。
    """
    ppl = calculate_direct_ppl(text)
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) == 0:
        return float("inf")
    comp = zlib.compress(raw)
    compression_ratio = len(comp) / max(1, len(raw))
    if compression_ratio <= 0:
        return float("inf")
    return float(ppl / compression_ratio)


def calculate_lowercase_score(text, max_length=512):
    """
    lowercase baseline：PPL(text) / PPL(lowercase(text))。
    """
    ppl_orig = calculate_direct_ppl(text)
    ppl_lower = calculate_direct_ppl(text.lower())
    if ppl_lower <= 0:
        return float("inf")
    return float(ppl_orig / ppl_lower)


def calculate_prefix_suffix_ppl(text, prefix_tokens=256, max_length=512):
    """
    Prefix->Suffix baseline 的近似实现：
    取后缀文本的直接 PPL 作为分数（越小越偏成员）。
    """
    tokens = text.split()
    if len(tokens) <= prefix_tokens + 1:
        return calculate_direct_ppl(text)
    suffix = " ".join(tokens[prefix_tokens:])
    return calculate_direct_ppl(suffix[: max_length * 6])


'''
# 示例主流程
long_text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It focuses on how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. Techniques used in NLP include tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation. Deep learning approaches have achieved significant success in recent years across many NLP tasks. Transformers models like BERT and GPT have pushed the state-of-the-art performance on benchmarks to near-human levels on certain tasks. However, challenges remain including handling ambiguity, understanding context, interpreting figurative language, and dealing with low-resource languages. NLP applications are widespread in modern technology: search engines use it to understand queries, chatbots employ it for conversation, email clients utilize it for spam filtering, and voice assistants depend on it for speech recognition and response generation. As datasets grow larger and models become more complex, efficient processing of long documents remains a critical challenge in the field."
augmented = mask_and_fill(long_text, 0.15, 512)
print(f"原始文本: {long_text}")
print(f"生成文本: {augmented}")

# 计算simhash相似度
hamming_distance = simhash_similarity(long_text, augmented)
print(f"SimHash 汉明距离: {hamming_distance}")

# 在主流程中调用困惑度估算
# masked_tokens, mask_spans = mask_blocks(long_text.split(), 0.15)
# augmented = " ".join(fill_masks(masked_tokens, mask_spans, 512))
ppl = calculate_direct_ppl(augmented)
print(f"困惑度估算: {ppl}")'''
