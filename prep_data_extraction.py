import pandas as pd
import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
tqdm.tqdm.pandas()
import json
from pyvi import ViTokenizer
from python_rdrsegmenter import load_segmenter
from argparse import ArgumentParser
from typing import List
from multiprocessing import Pool
import numpy as np
from data_helper.encode_instance import InstanceEncoder

parser = ArgumentParser()
parser.add_argument("--model_type",type=str, default='bert')
parser.add_argument("--negative_sampling",type=str, default='true')
args = parser.parse_args()
segmenter = load_segmenter()



def load_data(data_path):
    data = json.load(open(data_path, "r", encoding="utf-8")).get("data")
    d = {"question": [], "context": [], "answer": [], "start_char_position": []}
    for para in tqdm.tqdm(data):
        for p in para["paragraphs"]:
            context = p["context"]
            qas = p["qas"]
            for qa in qas:
                question = qa["question"]
                answers = []
                positions = []
                for a in qa["answers"]:
                    positions.append(a["answer_start"])
                    answers.append(a["text"])
                d["question"].append(question)
                d["answer"].append(answers)
                d["start_char_position"].append(positions)
                d["context"].append(context)
    df = pd.DataFrame.from_dict(d)
    return df

train_df = load_data('./data/train_ViQuAD.json')
dev_df = load_data('./data/dev_ViQuAD.json')
test_df = load_data('./data/test_ViQuAD.json')



def normalize_special_char(question: str, context: str, answers: List[str], start_char_positions: List[int]):
    # answer = ViTokenizer.tokenize(answer).replace("_", " ")
    # Cách normalize answer phải giống như cách normalize context
    # để đảm bảo sau normalize thì answer vẫn tồn tại trong context
    # vì vậy sẽ normalize từng token một, giống như context
    # và để đảm bảo không phải replace "_" bằng " "
    # tránh mất đi "_" vốn có trong câu
    normalized_question = []
    for token in question.split():
        normalized_question.append(ViTokenizer.tokenize(token))
    normalized_question = " ".join(normalized_question)

    normalized_answers = []
    for answer in answers:
        normalized_answer = []
        for token in answer.split():
            normalized_answer.append(ViTokenizer.tokenize(token))
        normalized_answer = " ".join(normalized_answer)
        normalized_answers.append(normalized_answer)

    # xác định vị trí token bắt đầu của câu trả lời
    new_start_char_positions = [0]*len(start_char_positions)
    start_token_positions = [len(context[:start_char_position+1].split()) - 1 for start_char_position in start_char_positions]

    tokens = context.split()
    normalized_tokens = []
    for i, token in enumerate(tokens):
        token = ViTokenizer.tokenize(token).replace("_", " ")
        normalized_tokens.append(token)
        n = len(token)
        for j in range(len(start_token_positions)):
            if i < start_token_positions[j]:
                new_start_char_positions[j] += n + 1 # space
            elif i == start_token_positions[j]:
                for tok in token:
                    if answers[j].startswith(tok):
                        break
                    new_start_char_positions[j] += 1

                
    return normalized_question, " ".join(normalized_tokens), normalized_answers, new_start_char_positions

def segment(text):
    segmented_text = segmenter.tokenize(text)
    # assert len(segmented_text) == len(text)
    if len(segmented_text) != len(text) and segmented_text.startswith("_") or segmented_text.endswith("_"):
        segmented_text = segmented_text.strip("_")
    assert len(segmented_text) == len(text)
    return segmented_text

def positions(context, answers, start_char_positions):
    start_token_positions = []
    end_token_positions = []
    for answer, start_char_position in zip(answers, start_char_positions):
        start_token_position = len(context[:start_char_position+1].split()) - 1
        end_token_position = start_token_position + len(context[start_char_position: start_char_position+len(answer)].split()) - 1

        start_token_positions.append(start_token_position)
        end_token_positions.append(end_token_position)

    return start_token_positions, end_token_positions




def preprocess_data(df, model_type='phobert'):
    df["results"] = df.progress_apply(lambda x: normalize_special_char(x.question, x.context, x.answer, x.start_char_position), axis=1)
    df.question = df.results.apply(lambda x: x[0])
    df.context = df.results.apply(lambda x: x[1])
    df.answer = df.results.apply(lambda x: x[2])
    df.start_char_position = df.results.apply(lambda x: x[3])
    if model_type=='phobert':
        df.question = df.question.progress_apply(lambda x: segment(x))
        df.context = df.context.progress_apply(lambda x: segment(x))
        df.answer = df.answer.progress_apply(lambda x: [segment(e) for e in x])
    df["results"] = df.progress_apply(lambda x: positions(x.context, x.answer, x.start_char_position), axis=1)
    df["start_token_position"] = df.results.progress_apply(lambda x: x[0])
    df["end_token_position"] = df.results.progress_apply(lambda x: x[1])
    del df["results"]
    return df


train_df = preprocess_data(train_df, model_type=args.model_type)
dev_df = preprocess_data(dev_df, model_type=args.model_type)
test_df = preprocess_data(test_df, model_type=args.model_type)


# Negative Sampling
if args.negative_sampling=='true':
    from rank_bm25 import BM25Okapi
    import string

    def normalize(text):
        text = text.lower()
        for e in string.punctuation:
            if e != "_":
                text = text.replace(e, " ")
        return " ".join(text.split())
        # text = text.translate(str.maketrans(" ", " ", string.punctuation))

    def get_corpus(contexts):
        d = {}
        corpus = []
        for context in tqdm.tqdm(contexts):
            e = normalize(context)
            d[e] = context
            corpus.append(e.split())
        return d, corpus
    full_df = pd.concat([train_df, dev_df,test_df])
    d, corpus = get_corpus(full_df.drop_duplicates(subset=["context"]).context.values)
    bm25 = BM25Okapi(corpus)


    def get_similar_context(context: str, answers: List[str], n: int=10):
        similar_contexts = bm25.get_top_n(normalize(context).split(), corpus, n=len(corpus))
        results = []
        n = len(answers)*n
        count = 0
        for e in similar_contexts[1:]:
            e = " ".join(e)
            if not any(normalize(a) in e for a in answers):
                results.append(d[e])
                count += 1
                if count == n:
                    return results
        return results


    def parallelize_df(df, func, n_cores=12):
        df_split = np.array_split(df, n_cores)
        with Pool(n_cores) as pool:
            df = pd.concat(pool.map(func, df_split))
    
        return df

    def get_negative_context(df):
        df["negative_context"] = df.progress_apply(lambda x: get_similar_context(x.context, x.answer), axis=1)
        return df

    train_df = parallelize_df(train_df, get_negative_context)
    dev_df = parallelize_df(dev_df, get_negative_context)

    def gen_negative_instance(df):
        d = {"question": [], "context": [], "start_token_position": [], "end_token_position": []}
        for question, context, negative_contexts, start_token_position, end_token_position in tqdm.tqdm(zip(df.question.values, df.context.values, df.negative_context.values, df.start_token_position.values, df.end_token_position.values)):
            d["question"].append(question)
            d["context"].append(context)
            d["start_token_position"].append(start_token_position)
            d["end_token_position"].append(end_token_position)
            n_pos = len(start_token_position)
            n = 1
            count = 0
            for e in negative_contexts:
                d["question"].append(question)
                d["context"].append(e)
                d["start_token_position"].append([-1]*n_pos)
                d["end_token_position"].append([-1]*n_pos)
                count += 1
                if count == n:
                    break
        df = pd.DataFrame.from_dict(d)
        return df

    train_df = gen_negative_instance(train_df)
    dev_df = gen_negative_instance(dev_df)

# test_df.to_csv("data/test_phobert_extraction.csv",index=False)
# raise ValueError("DONE")

## Encoding to word index
def encode(df, model_type='phobert',mode='train'):
    questions = df.question.values
    contexts=df.context.values
    if model_type == "phobert":
        tokenizer = AutoTokenizer.from_pretrained(
            'vinai/phobert-base', do_lower_case=False, remove_accents=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False, remove_accents=False)
        questions = [q.replace('_',' ') for q in questions]
        contexts =  [c.replace('_',' ') for c in contexts]

    instance_encoder = InstanceEncoder(tokenizer)
    instances = instance_encoder.encode(
        questions=questions,
        contexts=contexts,
        start_positions=df.start_token_position.values,
        end_positions=df.end_token_position.values,
        training_mode= mode=='train'
    )

    return instances

train_instances = encode(train_df, args.model_type, 'train')
dev_instances = encode(dev_df, args.model_type, 'dev')
test_instances = encode(test_df, args.model_type, 'test')


if args.negative_sampling=='true':
    with open(f"data/train_{args.model_type}_extraction_ids.txt",'w+') as f:
        for e in train_instances:
            f.write(f"{str(e)}\n")
    with open(f"data/dev_{args.model_type}_extraction_ids.txt",'w+') as f:
        for e in dev_instances:
            f.write(f"{str(e)}\n")

else:
    with open(f"data/train_{args.model_type}_extraction_ids_no_negative.txt",'w+') as f:
        for e in train_instances:
            f.write(f"{str(e)}\n")

    with open(f"data/dev_{args.model_type}_extraction_ids_no_negative.txt",'w+') as f:
        for e in dev_instances:
            f.write(f"{str(e)}\n")

    with open(f"data/test_{args.model_type}_extraction_ids_no_negative.txt",'w+') as f:
        for e in test_instances:
            f.write(f"{str(e)}\n")

