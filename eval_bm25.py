import pandas as pd
from models.retrieval_model import RetrievalDataset
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
import string

train_dataset = RetrievalDataset("data/train_retrieval.json")
dev_dataset = RetrievalDataset("data/dev_retrieval.json")
test_dataset = RetrievalDataset("data/test_retrieval.json")
global_passages = []
seen_passage_ids = set()
for dataset in [train_dataset, dev_dataset, test_dataset]:
    for context, passage_id in zip(dataset.contexts, dataset.context_ids):
        if passage_id not in seen_passage_ids:
            global_passages.append({"text": context, "id": passage_id})
            seen_passage_ids.add(passage_id)
global_passage_df = pd.DataFrame(global_passages)
bm25 = BM25Okapi(global_passage_df.text.map(lambda x: ViTokenizer.tokenize(x).replace("_", ' ').translate(str.maketrans('', '', string.punctuation)).lower().split()).tolist())
result = defaultdict(list)
for question, passage_id in tqdm(zip(dev_dataset.questions, dev_dataset.context_ids)):
    question_query = ViTokenizer.tokenize(question).replace("_",' ').translate(str.maketrans('', '', string.punctuation)).lower().split()
    for k in [1,5, 10, 20, 100]:
        top_k_bm25_passages = bm25.get_top_n(question_query, global_passages, k)
        result[f'top_{k}_accuracy'].append(
                passage_id in map(lambda x: x['id'],top_k_bm25_passages))

eval_str = f"{tabulate(pd.DataFrame(result).mean().to_frame().T, headers='keys', tablefmt='psql', showindex=False)}"
print(eval_str)