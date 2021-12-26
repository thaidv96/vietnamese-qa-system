from pyvi import ViTokenizer
from data_helper.custom_dataset import CustomDataset
from data_helper.encode_instance import InstanceEncoder
from models.extraction_model import RobertaForQuestionAnswering, BertForQuestionAnswering
import torch
from typing import Text, List
from python_rdrsegmenter import load_segmenter
from transformers import AutoTokenizer, WordpieceTokenizer
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference



class Predictor:
    def __init__(self, checkpoint_path, device: Text = 'cuda:0'):
        self.device = device
        if 'phobert' in checkpoint_path:
            self.model_type_name = 'phobert'
        else:
            self.model_type_name = 'bert-base'
        if self.model_type_name == 'bert-base':
            self.model = BertForQuestionAnswering.from_pretrained(
                checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-cased")
        else:
            self.model = RobertaForQuestionAnswering.from_pretrained(checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "vinai/phobert-base")

        self.model.eval()
        self.model.to(device)
        self.instance_encoder = InstanceEncoder(self.tokenizer)
        self.maxlen = 256

        if self.model_type_name == "phobert":
            self.segmenter = load_segmenter()

    def predict(self, question: str, context: str):
        if isinstance(question, str):
            question = [ViTokenizer.tokenize(question).replace("_", ' ')]
        if isinstance(context, str):
            context = [ViTokenizer.tokenize(context).replace("_", ' ')]

        if self.model_type_name == "phobert":
            question = [self.segmenter.tokenize(e) for e in question]
            context = [self.segmenter.tokenize(e) for e in context]

        results = []

        data = self.instance_encoder.encode(question, context)
        data = CustomDataset(
            data,
            maxlen=self.maxlen,
            pad_token_id=self.tokenizer.pad_token_id
        )
        dataloader = DataLoader(data, batch_size=1)
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, mask, token_type_ids = batch
            # return input_ids
            with torch.no_grad():
                # if self.model_type_name == 'bert-base':
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                # else:
                #     outputs = self.model(
                #         input_ids=input_ids,
                #         attention_mask=mask,
                #     )

                _, start_logits, end_logits = outputs
                start_logits = start_logits.detach().cpu().numpy()
                end_logits = end_logits.detach().cpu().numpy()

                start_positions = start_logits.argmax(axis=-1)  # List[int]
                end_masks = np.repeat([range(end_logits.shape[1])],end_logits.shape[0],axis=0) >= start_positions.reshape(-1,1)
                end_positions = (end_logits*end_masks).argmax(axis=-1)  # List[int]

                start_probs = np.exp(start_logits)/np.exp(start_logits).sum(axis=1)
                end_probs = np.exp(end_logits)/np.exp(end_logits).sum(axis=1)
                # print("START PROBS",start_probs.shape,start_probs)


                for input_id, spos, epos,sprob, eprob in zip(input_ids, start_positions, end_positions,start_probs, end_probs):
                    score = sprob[spos] + eprob[epos]
                    words, spos, epos = self.convert_word_piece_position(
                        input_id, spos, epos)

                    answer = " ".join(words).replace('_',' ')
                    
                    
                    results.append((answer, spos, epos,score))
                    # results.append(
                    #     (" ".join(self.tokenizer.convert_ids_to_tokens(input_id[spos:epos+1])), input_id[spos:epos+1], spos, epos))
        return results

    # -> [words, spos, epos]
    def convert_word_piece_position(self, word_piece_ids: List[int], spos: int, epos: int):
        """This method does:
            - Converts start_position and end_position to list of indices
            - Merge indices of word-pieces to indices of complete words

            Args:
            - word_piece_ids (List[int]): list of word-piece ids obtained by Bert tokenizer
            - spos (int): start word-piece position of the answer
            - epos (int): end word-piece position of the answer

            Returns:

            Notes:
            Word-piece id at spos and epos are the same and is cls_token_id
            means there is no answer in the context,
            so returns []
        """
        # words = self.tokenizer.decode(
        #     word_piece_ids,
        #     skip_special_tokens=False
        # )
        # print(words)

        word_pieces = self.tokenizer.convert_ids_to_tokens(
            word_piece_ids,
            skip_special_tokens=False
        )
        if self.model_type_name == 'phobert':
            while '@@' in word_pieces[epos] and epos < len(word_pieces) - 1:
                epos +=1
            while '@@' in word_pieces[spos-1] and spos > 0:
                spos -= 1
        else:
            while epos < len(word_pieces)-2 and '##' in word_pieces[epos+1]:
                epos += 1
            while '##' in word_pieces[spos] and spos >0:
                spos -=1


        word_pieces = word_pieces[spos:epos+1]
        words = self.merge_word_piece(word_pieces)
        return words, spos, epos
        # if spos == epos == 0:
        #     return words, -1, -1
        # else:
        #     for i, e in enumerate(word_pieces):
        #         if self.is_wordpiece(e):
        #             if i < spos:
        #                 spos -= 1
        #             elif i < epos:
        #                 epos -= 1
        #             else:
        #                 break
        #     return words, spos, epos

    def is_wordpiece(self, word: Text):
        if self.model_type_name == "phobert":
            return word.endswith("@@")

        return word.startswith("##")

    def merge_word_piece(self, word_pieces: List[str]):
        words = []
        if self.model_type_name == "phobert":
            next_wp = False
            for i, e in enumerate(word_pieces):
                if next_wp:
                    words[-1].append(e.strip("@@"))
                else:
                    words.append([e.strip("@@")])

                if e.endswith("@@"):
                    next_wp = True
                else:
                    next_wp = False
            return ["".join(e) for e in words]

        else:
            for i, e in enumerate(word_pieces):
                if e.startswith("##") and i > 0:
                    words[-1].append(e[2:])
                else:
                    words.append([e])
            return ["".join(e) for e in words]
