from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel
from models.retrieval_model import RetrievalModel, RetrievalDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from python_rdrsegmenter import load_segmenter
import torch
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from collections import defaultdict
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import string

segmenter = load_segmenter()
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
device = torch.device('cuda:2')

question_model = AutoModel.from_pretrained("checkpoint/question_model/")
ctx_model = AutoModel.from_pretrained("checkpoint/ctx_model/")
model = RetrievalModel(question_model, ctx_model)
model.to(device)
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
# bm25 = BM25Okapi(global_passage_df.text.map(
#     lambda x: ViTokenizer.tokenize(x).replace("_", ' ').split()).tolist())
with open("data/bm25.pkl",'rb') as f:
    bm25 = pickle.load(f)


# Get all passage representation:
# passage_embeddings = []
# for doc in tqdm(global_passages):

#     passage_inputs = tokenizer(segmenter.tokenize(doc['text']), padding=True,
#                                truncation=True, max_length=256, return_tensors='pt')
#     passage_inputs = {k: v.to(device) for k, v in passage_inputs.items()}
#     passage_embeddings.append(model.get_representation(
#         model.ctx_model, **passage_inputs, fix_model=True).to('cpu').detach().numpy())
# passage_embeddings = np.vstack(passage_embeddings)
with open("data/dpr.pkl",'rb') as f:
    passage_embeddings = pickle.load(f)


# Create Meta model dataset:
meta_data = []
for question, passage_id in tqdm(zip(dev_dataset.questions, dev_dataset.context_ids)):
    question_inputs = tokenizer(segmenter.tokenize(question), padding=True,
                                truncation=True, max_length=50, return_tensors='pt')
    question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
    question_embedding = model.get_representation(
        model.question_model, **question_inputs, fix_model=True).to('cpu').detach().numpy().squeeze()
    dpr_scores = question_embedding.dot(passage_embeddings.T)
    question_query = ViTokenizer.tokenize(question).replace("_", ' ').translate(str.maketrans('', '', string.punctuation)).lower().split()
    bm25_scores = bm25.get_scores(question_query)
    global_top_k_indices = set(bm25_scores.argsort(
    )[-100:][::-1].tolist() + dpr_scores.argsort()[-100:][::-1].tolist())
    for idx in global_top_k_indices:
        meta_r = {}
        meta_r['dpr_score'] = dpr_scores[idx]
        meta_r['label'] = passage_id == global_passage_df.iloc[idx].id
        meta_r['bm25_score'] = bm25_scores[idx]
        meta_data.append(meta_r)

meta_data_df = pd.DataFrame(meta_data)

clf = LogisticRegression()

meta_train_df, meta_test_df = train_test_split(meta_data_df)

X_train = meta_train_df[['dpr_score', 'bm25_score']].values
y_train = meta_train_df['label'].values
sampler = RandomUnderSampler('majority')
X_train, y_train = sampler.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

clf.fit(X_train, y_train)

joblib.dump(clf, 'checkpoint/meta_model/clf.pkl')
joblib.dump(scaler, 'checkpoint/meta_model/scaler.pkl')

# Final Evaluation on Test Dataset
result = defaultdict(list)
for question, passage_id in tqdm(zip(test_dataset.questions, test_dataset.context_ids)):
    question_inputs = tokenizer(segmenter.tokenize(question), padding=True,
                                truncation=True, max_length=50, return_tensors='pt')
    question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
    question_embedding = model.get_representation(
        model.question_model, **question_inputs, fix_model=True).to('cpu').detach().numpy().squeeze()
    dpr_scores = question_embedding.dot(passage_embeddings.T)
    question_query = ViTokenizer.tokenize(question).replace("_", ' ').translate(str.maketrans('', '', string.punctuation)).lower().split()
    bm25_scores = bm25.get_scores(question_query)
    global_top_k_indices = list(set(bm25_scores.argsort(
    )[-100:][::-1].tolist() + dpr_scores.argsort()[-100:][::-1].tolist()))
    inp = np.hstack(
        [dpr_scores[global_top_k_indices].reshape(-1, 1), bm25_scores[global_top_k_indices].reshape(-1, 1)])
    normalized_inp = scaler.transform(inp)
    global_final_scores = clf.predict_proba(normalized_inp)[:, 1]
    arg_sort_global_final_score = np.array(global_final_scores).argsort()
    for k in (1,5, 10, 20, 100):
        final_indices = np.array(global_top_k_indices)[
            arg_sort_global_final_score[-k:][::-1]]
        result[f'top_{k}_accuracy'].append(
            passage_id in global_passage_df.iloc[final_indices].id.values)

print(pd.DataFrame(result).mean().to_frame())
