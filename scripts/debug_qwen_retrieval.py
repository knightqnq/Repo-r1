#!/usr/bin/env python3
import json
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================= 配置 =================
BASE_DIR = Path('/root/autodl-tmp/models/search-r1')
INDEX_DIR = BASE_DIR / 'data/indexes'
GOLD_FILE = BASE_DIR / 'data/swebench_gold_files.jsonl'
TASK_FILE = BASE_DIR / 'data/swe_bench_lite/swe_lite_test.jsonl'
EMBEDDING_PATH = "/root/autodl-tmp/repor1/base_model/Qwen3-Embedding-8B"

# ================= 加载模型 =================
print(f" Loading Qwen... ")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBEDDING_PATH, trust_remote_code=True, torch_dtype=torch.float16)
model.to(device)
model.eval()

def get_embeddings(texts, batch_size=4):
    """
    Qwen Embedding 官方推荐通常是不带 instruction 的直接编码，
    或者 EOS token pooling。用最稳健的 mean pooling 
    """
    all_embs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        with torch.no_grad():
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            
            # 尝试 Mean Pooling (通常比 Last Token 更稳健)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # 归一化
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embs.append(embeddings.cpu().numpy())
            
    return np.concatenate(all_embs, axis=0)

# ================= 测试单个 Case =================
# 选一个经典的 Django Case
TARGET_ID = "django__django-11001" 

print(f"\n Debugging Case: {TARGET_ID}")

# 1. 获取 Gold Answer
gold_files = set()
with open(GOLD_FILE) as f:
    for line in f:
        obj = json.loads(line.strip())
        if obj['instance_id'] == TARGET_ID:
            gold_files = set(obj['bug_files'])
            break
print(f" Target Files: {gold_files}")

# 2. 获取 Query
query = ""
with open(TASK_FILE) as f:
    for line in f:
        obj = json.loads(line.strip())
        if obj['instance_id'] == TARGET_ID:
            query = obj['problem_statement']
            break
print(f" Query Length: {len(query)} chars")

# 3. 读取所有文件路径 (从 Meta 文件)
meta_file = INDEX_DIR / f'{TARGET_ID}_meta.jsonl'
file_paths = []
if meta_file.exists():
    with open(meta_file) as f:
        for line in f:
            file_paths.append(json.loads(line.strip())['file_path'])
    print(f"Total Files in Repo: {len(file_paths)}")
else:
    print(" Meta file not found! 无法读取代码库文件列表。")
    exit(1)

# 4. 现场构建索引 (只取前 100 个文件 + 黄金文件，为了速度)
# 为了验证效果，我们需要确保 Gold File 在被索引的列表里
candidates = list(gold_files)
# 补充一些干扰项
remaining = [f for f in file_paths if f not in gold_files]
candidates.extend(remaining[:100]) 

print(f" Re-Embedding {len(candidates)} files on-the-fly...")

# 伪造代码内容 (因为我们没有原始代码库，只能用文件名代替内容进行测试)
# 注意：这只是为了验证 Retrieve 流程！真实的 Retrieve 需要代码内容。
# 如果想验证文件名检索，就只Embed文件名。
# 为了模拟真实度，我们假设： Content = Filename + " some python code context"
candidate_contents = [f"File: {f}\nContent: This is the source code for {f.split('/')[-1]}..." for f in candidates]

# 生成文档向量
doc_embs = get_embeddings(candidate_contents)

# 生成查询向量
query_emb = get_embeddings([query])

# 5. 检索
d = doc_embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(doc_embs)

D, I = index.search(query_emb, 10)

print("\nRESULTS:")
for rank, idx in enumerate(I[0]):
    retrieved_file = candidates[idx]
    score = D[0][rank]
    hit = " HIT" if retrieved_file in gold_files else "❌"
    print(f"{rank+1}. [{score:.4f}] {retrieved_file} {hit}")