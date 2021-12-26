from copy import deepcopy
from .retriever import Retriever
import pickle
from ..common import global_passages
from pyvi import ViTokenizer

class BM25Retriever(Retriever):
    def __init__(self):
        with open("data/bm25.pkl",'rb') as f:
            self.bm25 = pickle.load(f)
    
    def query(self, question, top_k=20, return_all_scores=False):
        question_query = ViTokenizer.tokenize(question).replace("_",' ').split()
        scores = self.bm25.get_scores(question_query)
        if return_all_scores:
            return scores
        argsort_scores = scores.argsort()
        top_k_indices = argsort_scores[-top_k:][::-1]
        res = []
        for idx in top_k_indices:
            r = deepcopy(global_passages[idx])
            r['score'] = scores[idx]
            res.append(r)
        return res