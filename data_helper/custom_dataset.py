import torch
from typing import Text, List, Tuple, Iterator
import os
import sys
sys.path.insert(1, os.getcwd())


class CustomDataset:
    def __init__(
        self,
        instances: Iterator[Tuple[List[int]]],
        maxlen: int,
        pad_token_id: int,
        size: int = -1
    ):  # -> None
        """[summary]
        This class is used instead of torch TensorDataset

        Args:
            instances (Iterator[Tuple[List[int]]]): Output of method `encode` in class InstanceEncoder
            maxlen (int): Max sequence length
            pad_token_id (int): Value which is added to sequence to pad sequence length
        """
        # for memory ... `instances`` should be loaded from .txt file
        # instances = file.read().splitlines()
        self.instances = instances
        self.maxlen = maxlen
        self.pad_token_id = pad_token_id
        self.size = size

    def __getitem__(self, index):
        instance = self.instances[index]
        if isinstance(instance, str):
            instance = eval(instance)

        question_context = torch.LongTensor(self.pad_sequence_length(
            instance[0], maxlen=self.maxlen, pad_token_id=self.pad_token_id))
        mask = torch.LongTensor([int(e != self.pad_token_id)
                                for e in question_context])
        token_type_ids = torch.LongTensor(self.pad_sequence_length(
            instance[1], maxlen=self.maxlen, pad_token_id=1))
        if len(instance) > 2:
            spos = torch.LongTensor(instance[2])
            epos = torch.LongTensor(instance[3])
            return (question_context, mask, token_type_ids, spos, epos)
        return (question_context, mask, token_type_ids)

    def __len__(self):
        if self.size < 0:
            return len(self.instances)
        return self.size

    @staticmethod
    def pad_sequence_length(sequence: List[int], maxlen: int, pad_token_id: int):
        if len(sequence) > maxlen:
            return sequence[:maxlen]
        elif len(sequence) < maxlen:
            sequence.extend([pad_token_id]*(maxlen - len(sequence)))
        return sequence
