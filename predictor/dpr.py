import sys
import os
sys.path.append(os.getcwd())

import pickle
import torch
from tqdm import tqdm
from models.retrieval_model import RetrievalModel
import numpy as np
from python_rdrsegmenter import load_segmenter
from transformers import AutoTokenizer, AutoModel
from predictor.common import global_passages

if __name__ == '__main__':
    device = torch.device('cuda:2')
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    segmenter = load_segmenter()
    question_model = AutoModel.from_pretrained("checkpoint/question_model/")
    ctx_model = AutoModel.from_pretrained("checkpoint/ctx_model/")
    model = RetrievalModel(question_model, ctx_model)
    model.to(device)

    passage_embeddings = []
    for doc in tqdm(global_passages):

        passage_inputs = tokenizer(segmenter.tokenize(doc['text']), padding=True,
                                   truncation=True, max_length=256, return_tensors='pt')
        passage_inputs = {k: v.to(device) for k, v in passage_inputs.items()}
        passage_embeddings.append(model.get_representation(
            model.ctx_model, **passage_inputs, fix_model=True).to('cpu').detach().numpy())
    passage_embeddings = np.vstack(passage_embeddings)
    with open("data/dpr.pkl", 'wb') as f:
        pickle.dump(passage_embeddings, f)
