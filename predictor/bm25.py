from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from common import global_passage_df
import pickle
import string

if __name__ == "__main__":
    bm25 = BM25Okapi(global_passage_df.text.map(
    lambda x: ViTokenizer.tokenize(x).replace("_", ' ').translate(str.maketrans('', '', string.punctuation)).lower().split()).tolist())
    with open("data/bm25.pkl", 'wb') as f:
        pickle.dump(bm25, f)
