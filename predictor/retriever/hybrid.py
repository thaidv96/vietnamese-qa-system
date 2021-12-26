from copy import deepcopy
from .retriever import Retriever
from .bm25 import BM25Retriever
from ..common import global_passages
from .semantics import SemanticsRetriever
import numpy as np
import joblib

meta_model_scaler = joblib.load('checkpoint/meta_model/scaler.pkl')
meta_model_clf = joblib.load('checkpoint/meta_model/clf.pkl')

class HybridRetriever(Retriever):
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.semantics_retriever = SemanticsRetriever()

    def query(self, question, top_k=20):
        bm25_scores = self.bm25_retriever.query(question, return_all_scores=True)
        dpr_scores = self.semantics_retriever.query(question, return_all_scores=True)
        global_top_k_indices = list(set(bm25_scores.argsort()[-100:][::-1].tolist() + dpr_scores.argsort()[-100:][::-1].tolist()))
        inp = np.hstack(
            [dpr_scores[global_top_k_indices].reshape(-1, 1), bm25_scores[global_top_k_indices].reshape(-1, 1)])
        normalized_inp = meta_model_scaler.transform(inp)
        global_final_scores = meta_model_clf.predict_proba(normalized_inp)[:, 1]
        arg_sort_global_final_score = np.array(global_final_scores).argsort()
        final_indices = np.array(global_top_k_indices)[
            arg_sort_global_final_score[-top_k:][::-1]]
        res = []
        for idx in final_indices:
            r = deepcopy(global_passages[idx])
            r['score'] = global_final_scores[global_top_k_indices.index(idx)]
            res.append(r)
        return res
