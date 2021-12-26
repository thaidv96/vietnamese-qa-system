from models.retrieval_model import RetrievalDataset
import pandas as pd
train_dataset = RetrievalDataset("data/train_retrieval.json")
dev_dataset = RetrievalDataset("data/dev_retrieval.json")
test_dataset = RetrievalDataset("data/test_retrieval.json")

global_passages = []
seen_passage_ids = set()
for dataset in [train_dataset, dev_dataset, test_dataset]:
    for context, passage_id in zip(dataset.original_contexts, dataset.context_ids):
        if passage_id not in seen_passage_ids:
            global_passages.append({"text": context, "id": passage_id})
            seen_passage_ids.add(passage_id)
global_passage_df = pd.DataFrame(global_passages)
