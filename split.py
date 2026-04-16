import nltk
from nltk.tokenize import sent_tokenize

def split_document(doc, max_tokens):
    """
    分割长文本为完整句子组成的序列（每块 ≤ max_tokens）
    """
    sentences = sent_tokenize(doc)
    chunks, current_chunk = [], []
    current_length = 0

    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        if current_length + len(tokens) > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(sent)
        current_length += len(tokens)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

'''
d = "Hello! How are you?"
long_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. 
It focuses on how to program computers to process and analyze large amounts of natural language data. 
The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. 
Techniques used in NLP include tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation. 
Deep learning approaches have achieved significant success in recent years across many NLP tasks. 
Transformers models like BERT and GPT have pushed the state-of-the-art performance on benchmarks to near-human levels on certain tasks. 
However, challenges remain including handling ambiguity, understanding context, interpreting figurative language, and dealing with low-resource languages. 
NLP applications are widespread in modern technology: search engines use it to understand queries, chatbots employ it for conversation, email clients utilize it for spam filtering, and voice assistants depend on it for speech recognition and response generation. 
As datasets grow larger and models become more complex, efficient processing of long documents remains a critical challenge in the field.
""" * 8

# 显示字符数
print(f"文本总字符数: {len(long_text)}")
print(f"文本总句数: {len(sent_tokenize(long_text))}\n")

# 使用函数进行分割
chunks = split_document(long_text, max_tokens=512)
tokens  = chunks[0].split()

# 验证结果
print(f"结果: {chunks}")
print(tokens)'''
