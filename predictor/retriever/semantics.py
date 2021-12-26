from copy import deepcopy
from .retriever import Retriever
import pickle
from ..common import global_passages
from python_rdrsegmenter import load_segmenter
from transformers import AutoModel, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

segmenter = load_segmenter()
device = torch.device('cuda:1')

class SemanticsRetriever(Retriever):
    def __init__(self):
        self.question_model = AutoModel.from_pretrained("checkpoint/question_model")
        self.question_model.to(device)
        self.question_model.eval()
        with open("data/dpr.pkl",'rb') as f:
            self.passage_embeddings = pickle.load(f)
     
    def query(self, question, top_k=20, return_all_scores=False):
        question_inputs = tokenizer(segmenter.tokenize(question), padding=True,
                                    truncation=True, max_length=50, return_tensors='pt')
        question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
        with torch.no_grad():
            question_embedding = self.question_model(**question_inputs)[1].to('cpu').detach().numpy().squeeze()
        sim_scores = question_embedding.dot(self.passage_embeddings.T)
        if return_all_scores:
            return sim_scores
        arg_sort_global_scores = sim_scores.argsort()
        global_top_k_indices = arg_sort_global_scores[-top_k:][::-1]
        res = []
        for idx in global_top_k_indices:
            r = deepcopy(global_passages[idx])
            r['score'] = sim_scores[idx]
            res.append(r)
        
        return res

