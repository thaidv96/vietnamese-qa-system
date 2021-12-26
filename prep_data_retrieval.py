from turtle import clone
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
import json
from itertools import chain
from tqdm import tqdm
from copy import deepcopy
dataset_folder = './data'
corpus = {}
tokenized_corpus = {}
dataset = {}
for dataset_type in ['train','dev','test']:
    with open(f"{dataset_folder}/{dataset_type}_ViQuAD.json") as f:
        dataset[dataset_type] = json.load(f)
    corpus[dataset_type] = [{"text":paragraph['context'], "title": article['title']} for article in dataset[dataset_type]['data'] for paragraph in article['paragraphs']]
    tokenized_corpus[dataset_type] = [ViTokenizer.tokenize(doc['text']).replace("_",' ').split() for doc in corpus[dataset_type]]

corpus = list(chain(*corpus.values()))
context_dataset = list(chain(*tokenized_corpus.values()))
bm25 = BM25Okapi(context_dataset)

print("Start Construct Retrieval Dataset for ViQuAD")
k = 50
retrieval_dataset = {}
for dataset_type in tqdm(['train','dev','test']):
    retrieval_dataset[dataset_type] = []
    for article in tqdm(dataset[dataset_type]['data'], leave=False):
        for paragraph in article['paragraphs']:
            for qas in paragraph['qas']:
                instance = {'positive_ctxs':[{'text': paragraph['context'], 'title': article['title'], 'passage_id': qas['id'].rsplit("_",1)[0]}]} 
                instance['question'] = qas['question']
                # tokenized_question = ViTokenizer.tokenize(qas['question']).replace("_",' ').split()
                # top_k_bm25_docs = bm25.get_top_n(tokenized_question, corpus,n=k)
                # answers = qas['answers']
                # for doc in top_k_bm25_docs:
                #     does_contain_answer = False
                #     for answer in answers:
                #         if answer['text'] in doc['text']:
                #             cloned_doc =deepcopy(doc)
                #             cloned_doc['answer'] = answer['text']
                #             instance['positive_ctxs'].append(cloned_doc)
                #             break
                retrieval_dataset[dataset_type].append(instance)

    with open(f"data/{dataset_type}_retrieval.json",'w+', encoding='utf8') as f:
        json.dump(retrieval_dataset[dataset_type], f, ensure_ascii=False)