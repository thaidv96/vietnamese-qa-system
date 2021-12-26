from predictor.retriever import SemanticsRetriever, BM25Retriever, bm25
from models.retrieval_model import RetrievalDataset
import numpy as np
from tqdm import tqdm

bm25_retriever = BM25Retriever()
semantics_retriever = SemanticsRetriever()

test_dataset = RetrievalDataset("data/test_retrieval.json")

bm25_covered_by_semantics = []
semantics_covered_by_bm25 = []

for idx in tqdm(range(len(test_dataset))):
    item = test_dataset[idx]
    question = item['question']
    bm25_result = bm25_retriever.query(question, top_k=20)
    bm25_result = [r['id'] for r in bm25_result]
    semantics_result = semantics_retriever.query(question, top_k=20)
    semantics_result = [r['id'] for r in semantics_result]
    if item['context_id'] not in bm25_result:
        bm25_covered_by_semantics.append(item['context_id'] in semantics_result)
    if item['context_id'] not in semantics_result:
        semantics_covered_by_bm25.append(item['context_id'] in bm25_result)

print("bm25_covered_by_semantics RATIO", np.mean(bm25_covered_by_semantics))
print("semantics_covered_by_bm25 RATIO", np.mean(semantics_covered_by_bm25))