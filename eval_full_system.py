from argparse import ArgumentParser
from pyvi import ViTokenizer
import json
from predictor import ReaderPredictor, QASystem
import numpy as np
import difflib
from predictor.retriever import BM25Retriever, SemanticsRetriever, HybridRetriever
from tqdm import tqdm

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return s1[pos_a:pos_a+size]


with open("data/test_ViQuAD.json") as f:
    test_data = json.load(f)['data']


def f1_score(gt_answer, pred_answer):
    tokenized_gt_answer = ViTokenizer.tokenize(gt_answer).replace("_", ' ')
    overlap_answer = get_overlap(tokenized_gt_answer, pred_answer)
    shared_tokens = overlap_answer.split()
    pred_tokens = pred_answer.split()
    if len(pred_tokens) == 0:
        return 0
    gt_tokens = tokenized_gt_answer.split()

    precision_score = len(shared_tokens)/len(pred_tokens)
    recall_score = len(shared_tokens)/len(gt_tokens)
    if precision_score + recall_score == 0:
        return 0
    else:
        return 2*precision_score*recall_score/(precision_score+recall_score)

norm_mapper = np.load('predictor/mapping_versions_of_words.npy', allow_pickle=True).item()

def em_score(gt_answer, pred_answer):
    tokenized_gt_answer = ViTokenizer.tokenize(gt_answer).replace("_", ' ')
    gt_tokens = [norm_mapper.get(i,i) for i in tokenized_gt_answer.split()]
    pred_tokens = [norm_mapper.get(i,i) for i in pred_answer.split()]


    return gt_tokens == pred_tokens


def eval_system(system: QASystem):
    f1s = []
    ems = []
    for article in tqdm(test_data):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                pred_answer = system.predict(qa['question'],args.retriever_size)
                max_f1_score = 0
                max_em_score = 0
                for answer in qa['answers']:
                    f1 = f1_score(answer['text'], pred_answer)
                    if f1 > max_f1_score:
                        max_f1_score = f1
                    em = em_score(answer['text'], pred_answer)
                    if em > max_em_score:
                        max_em_score = em
                f1s.append(max_f1_score)
                ems.append(max_em_score)
                if max_em_score:
                    with open("correct_questions.txt", 'a+') as f:
                        f.write(f"{qa['question']}\n")
    return {'f1_score': np.mean(f1s), 'em_score': np.mean(ems)}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--retriever_type", type=str)
    parser.add_argument('--reader_checkpoint_path', type=str)
    parser.add_argument('--retriever_size', type=int, default=20)

    args = parser.parse_args()
    if args.retriever_type == 'bm25':
        retriever = BM25Retriever()
    elif args.retriever_type == 'semantics':
        retriever = SemanticsRetriever()
    else:
        retriever = HybridRetriever()
    reader = ReaderPredictor(args.reader_checkpoint_path)
    system = QASystem(retriever, reader)
    res = eval_system(system)
    print(res)
