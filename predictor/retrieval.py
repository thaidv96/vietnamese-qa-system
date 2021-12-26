from .common import global_passages
import pickle
import joblib
import numpy as np
from python_rdrsegmenter import load_segmenter
from transformers import AutoTokenizer, AutoModel
import torch
from models.retrieval_model import RetrievalModel
from pyvi import ViTokenizer
import sys
import os

sys.path.append(os.getcwd())
segmenter = load_segmenter()
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
device = torch.device('cuda:2')
question_model = AutoModel.from_pretrained("checkpoint/question_model/")
ctx_model = AutoModel.from_pretrained("checkpoint/ctx_model/")
model = RetrievalModel(question_model, ctx_model)
model.to(device)

with open("data/dpr.pkl", 'rb') as f:
    passage_embeddings = pickle.load(f)
with open('data/bm25.pkl', 'rb') as f:
    bm25 = pickle.load(f)

meta_model_scaler = joblib.load('checkpoint/meta_model/scaler.pkl')
meta_model_clf = joblib.load('checkpoint/meta_model/clf.pkl')


def get_relevant_passages(question, k=20):
    question_inputs = tokenizer(segmenter.tokenize(question), padding=True,
                                truncation=True, max_length=50, return_tensors='pt')
    question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
    question_embedding = model.get_representation(
        model.question_model, **question_inputs, fix_model=True).to('cpu').detach().numpy().squeeze()
    dpr_scores = question_embedding.dot(passage_embeddings.T)
    question_query = ViTokenizer.tokenize(question).replace("_", ' ').split()
    bm25_scores = bm25.get_scores(question_query)
    global_top_k_indices = list(set(bm25_scores.argsort(
    )[-100:][::-1].tolist() + dpr_scores.argsort()[-100:][::-1].tolist()))
    inp = np.hstack(
        [dpr_scores[global_top_k_indices].reshape(-1, 1), bm25_scores[global_top_k_indices].reshape(-1, 1)])
    normalized_inp = meta_model_scaler.transform(inp)
    global_final_scores = meta_model_clf.predict_proba(normalized_inp)[:, 1]
    arg_sort_global_final_score = np.array(global_final_scores).argsort()
    final_indices = np.array(global_top_k_indices)[
        arg_sort_global_final_score[-k:][::-1]]
    return [global_passages[i] for i in final_indices]
