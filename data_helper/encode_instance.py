from data_helper.encode_text import TextEncoder
from transformers import BertTokenizer
import tqdm
from typing import Text, List, Tuple
import os
import sys
sys.path.insert(1, os.getcwd())


class InstanceEncoder:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = TextEncoder(self.tokenizer)

    def encode(
        self,
        questions: List[str],
        contexts: List[str],
        start_positions: List[List[int]] = None,
        end_positions: List[List[int]] = None,
        training_mode: bool = False
    ):  # -> List[Tuple[List[int]]]
        encoded_instances = []
        if start_positions is not None and end_positions is not None:
            for question, context, spos, epos in tqdm.tqdm(zip(questions, contexts, start_positions, end_positions)):
                question = self.text_encoder.encode(
                    question,
                    add_cls_token=True,
                    add_sep_token=True
                )
                # Start position and end position will change
                # when context is encoded by tokenizer
                # because a token can be splitted into multi word-pieces
                tokens = context.split()
                # spos: List[int], epos: List[int]
                raw_spos = spos.copy()
                raw_epos = epos.copy()
                context = []
                for i, token in enumerate(tokens):
                    token = self.text_encoder.encode(
                        token, add_cls_token=False, add_sep_token=False)
                    context.extend(token)
                    # Update start and end position
                    for j in range(len(raw_spos)):
                        # start position is the FIRST WORD PIECE of a complete word
                        if i < raw_spos[j]:
                            spos[j] += len(token) - 1
                        # end position là word piece ĐẦU TIÊN của một từ
                        # thay vì word piece cuối cùng
                        # như vậy thì không phải tách dấu câu khỏi việc dính vào từ, ví dụ: hôm nay, -> hôm nay ,
                        # khi inference có kết quả ta sẽ nối các word pieces lại với nhau thành một từ hoàn chỉnh
                        if i < raw_epos[j]:
                            epos[j] += len(token) - 1
                context.append(self.tokenizer.sep_token_id)
                for j in range(len(spos)):
                    if spos[j] == -1:  # no answer
                        spos[j] = 0  # position 0 of [CLS]
                    else:
                        spos[j] += len(question)

                for j in range(len(epos)):
                    if epos[j] == -1:  # no answer
                        epos[j] = 0
                    else:
                        epos[j] += len(question)

                if training_mode:
                    spos = spos[:1]
                    epos = epos[:1]

                token_type_ids = len(question)*[0] + len(context)*[1]
                question.extend(context)
                encoded_instances.append(
                    (question, token_type_ids, spos, epos))
        else:
            for question, context in zip(questions, contexts):
                question = self.text_encoder.encode(
                    question, add_cls_token=True, add_sep_token=True)
                context = self.text_encoder.encode(
                    context, add_cls_token=False, add_sep_token=True)
                token_type_ids = len(question)*[0] + len(context)*[1]
                question.extend(context)
                encoded_instances.append((question, token_type_ids))
        return encoded_instances
