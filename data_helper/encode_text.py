import os
import sys
sys.path.insert(1, os.getcwd())

from typing import Text, List
from transformers import BertTokenizer


class TextEncoder:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: Text, add_cls_token: bool=False, add_sep_token: bool=False): # -> List[int]
        """This method does:
            - Convert each token in text to ids in vocab of self.tokenizer
            - Each token can be splitted into word-piece
        """
        if add_cls_token:
            text = f"{self.tokenizer.cls_token} {text}"
        if add_sep_token:
            text = f"{text} {self.tokenizer.sep_token}"

        return self.tokenizer.encode(text, add_special_tokens=False)