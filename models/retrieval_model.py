from torch import nn
import torch
from torch import tensor as T
from typing import List
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from iteration_utilities import unique_everseen
from pyvi import ViTokenizer

class RetrievalModel(nn.Module):
    def __init__(self, question_model: nn.Module, ctx_model: nn.Module):
        super(RetrievalModel, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model

    @staticmethod
    def get_representation(model: nn.Module, input_ids: T, token_type_ids: T, attention_mask: T, fix_model: bool = False) -> T:
        if fix_model:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,  attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = model(
                input_ids=input_ids,  attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs[1]

    def forward(self, question_inputs, context_inputs, context_ids):
        q_pooled_ouput = self.get_representation(
            self.question_model, **question_inputs)
        ctx_pooled_output = self.get_representation(
            self.ctx_model, **context_inputs)
        labels = []
        unique_context_ids = list(unique_everseen(context_ids))
        for id in context_ids:
            labels.append(unique_context_ids.index(id))
        loss = RetrievalLoss().calc(q_pooled_ouput, ctx_pooled_output, labels)
        return loss, q_pooled_ouput, ctx_pooled_output




def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


class RetrievalLoss(object):
    def calc(self, q_vector: T, ctx_vectors: T, positive_idx_per_question: list) -> T:
        sim_scores = dot_product_scores(q_vector, ctx_vectors)
        softmax_scores = F.log_softmax(sim_scores, dim=1)
        loss = F.nll_loss(softmax_scores, torch.tensor(
            positive_idx_per_question).to(softmax_scores.device), reduction='mean')
        return loss


class RetrievalDataset(Dataset):
    def __init__(self, data_path, sample_size=None):
        with open(data_path) as f:
            data = json.load(f)
        questions = [ViTokenizer.tokenize(r['question']).replace("_",' ') for r in data]
        contexts = [r['positive_ctxs'][0]['title'] + ' ' +
                    r['positive_ctxs'][0]['text'] for r in data]
        contexts = [ViTokenizer.tokenize(x).replace("_",' ') for x in contexts]
        context_ids = [r['positive_ctxs'][0]['passage_id'] for r in data]
        original_contexts =  [r['positive_ctxs'][0]['text'] for r in data]
        self.questions = questions
        self.contexts = contexts
        self.context_ids = context_ids
        self.sample_size = sample_size
        self.original_contexts = original_contexts

    def __getitem__(self, idx):
        item = {"question": self.questions[idx],
                "context": self.contexts[idx],
                "context_id": self.context_ids[idx]}
        return item

    def __len__(self):
        if self.sample_size:
            return self.sample_size
        return len(self.questions)
