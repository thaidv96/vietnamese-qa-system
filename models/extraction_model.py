from typing import Text, List, Union
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from transformers import BertPreTrainedModel, BertModel, RobertaModel, RobertaPreTrainedModel, BertTokenizer
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import tqdm
import numpy as np
from pathlib import Path
from typing import Text, List
import os

custom_config = {
    "label_smoothing": 0.1
}

class QAMetric:
    def __init__(self, true_indices: List[List[int]], pred_indices: List[int]) -> None:
        # true_indices and pred_indices if list indices of answer
        # eg: start_token = 2, end_token = 5 -> indices: [2, 3, 4, 5]
        self.true_indices = true_indices
        self.pred_indices = pred_indices

    def max_f1_score(self, only_positive: bool = False):
        return max([self.f1_score(e, self.pred_indices, only_positive=only_positive) for e in self.true_indices])

    def max_exact_match(self, only_positive: bool = False):
        return max([self.exact_match(e, self.pred_indices, only_positive=only_positive) for e in self.true_indices])

    @staticmethod
    def f1_score(true_indices: List[int], pred_indices: List[int], only_positive: bool = False):
        if len(true_indices) == 0:
            if only_positive:
                return np.nan
            else:
                return int(len(pred_indices) == 0)
        elif len(pred_indices) == 0:
            return int(len(true_indices) == 0)
        else:
            shared_tokens = set(true_indices) & set(pred_indices)
            precision_score = len(shared_tokens)/len(pred_indices)
            recall_score = len(shared_tokens)/len(true_indices)
            if precision_score + recall_score == 0:
                return 0
            else:
                return 2*precision_score*recall_score/(precision_score+recall_score)

    @staticmethod
    def exact_match(true_indices: List[int], pred_indices: List[int], only_positive: bool = False):
        if only_positive:
            if len(true_indices) != 0:
                return int(true_indices == pred_indices)
            return np.nan
        else:
            return int(true_indices == pred_indices)





class RobertaForQuestionAnswering(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Outputs gồm last_hidden_state, pooler_output, hidden_states,...
        # Lấy phần tử đầu tiên của outputs là lấy last_hidden_state
        # Giả sử input_ids có kích thước (batch_size, số tokens) là (64, 16)
        # thì sentence_output (last_hidden_state) có kích thước (64, 16, 768) - mỗi token là một vector 768 chiều (do sử dụng bert-base)
        sequence_output = outputs[0]

        # sequence_output (64, 16, 768) đi qua một lớp linear có kích thước là (768, 2)
        # được logits có kích thước là (64, 16, 2) - đến đây kết quả giống như các bài toán phân lớp tokens
        # nghĩa là lấy hidden state qua một linear layer để phân lớp 16 tokens vào 2 lớp

        # Tuy nhiên với bài toán này ta biến đổi một chút để đưa thành bài toán phân lớp 2 tokens START và END vào 16 lớp
        # bằng cách split để biến đổi thành 2 logits của START và END token
        # Khi đó start_logits có size là (64, 16, 1) và end_logits cũng có size là (64, 16, 1)
        # qua squeeze thành (64, 16) và (64, 16) (16 lớp)
        # và ta tính cross entropy loss như bình thường
        # loss trả về là tổng 2 losses của start_logits và end_logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(
                ignore_index=ignored_index, label_smoothing=custom_config.get("label_smoothing", 0))
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits


class BertForQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Outputs gồm last_hidden_state, pooler_output, hidden_states,...
        # Lấy phần tử đầu tiên của outputs là lấy last_hidden_state
        # Giả sử input_ids có kích thước (batch_size, số tokens) là (64, 16)
        # thì sentence_output (last_hidden_state) có kích thước (64, 16, 768) - mỗi token là một vector 768 chiều (do sử dụng bert-base)
        sequence_output = outputs[0]

        # sequence_output (64, 16, 768) đi qua một lớp linear có kích thước là (768, 2)
        # được logits có kích thước là (64, 16, 2) - đến đây kết quả giống như các bài toán phân lớp tokens
        # nghĩa là lấy hidden state qua một linear layer để phân lớp 16 tokens vào 2 lớp

        # Tuy nhiên với bài toán này ta biến đổi một chút để đưa thành bài toán phân lớp 2 tokens START và END vào 16 lớp
        # bằng cách split để biến đổi thành 2 logits của START và END token
        # Khi đó start_logits có size là (64, 16, 1) và end_logits cũng có size là (64, 16, 1)
        # qua squeeze thành (64, 16) và (64, 16) (16 lớp)
        # và ta tính cross entropy loss như bình thường
        # loss trả về là tổng 2 losses của start_logits và end_logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(
                ignore_index=ignored_index, label_smoothing=custom_config.get("label_smoothing", 0))
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # if not return_dict:
        #     output = (start_logits, end_logits) + outputs[2:]
        #     return ((total_loss,) + output) if total_loss is not None else output

        return total_loss, start_logits, end_logits

        # return QuestionAnsweringModelOutput(
        #     loss=total_loss,
        #     start_logits=start_logits,
        #     end_logits=end_logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )



class Trainer:
    def __init__(self, pretrained_model: BertForQuestionAnswering, tokenizer: BertTokenizer, device: Text, log_file:Text=None) -> None:
        self.device = device
        self.model = pretrained_model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.lr = 2e-5
        self.eps = 1e-8
        self.warmup = 0.1
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.lr, eps=self.eps)
        self.scheduler = None
        self.log_file = log_file

    def train(
        self,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader = None,
        n_epochs: int = 4,
        model_dir: Text = None,
        log_file: Text = None,
        eval_each_epoch: bool = True
    ):
        self.log_file = log_file

        n_batches = len(train_dataloader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.warmup, num_training_steps=n_epochs*n_batches)

        self.log(
            f"Training on {len(train_dataloader.dataset)} samples".capitalize())
        for epoch in range(1, n_epochs+1):
            self.log(f"Epoch {epoch}")
            self.train_epoch(train_dataloader)
            self.save_model(os.path.join(model_dir, f"epoch_{epoch}"))
            if eval_each_epoch:
                self.log(f"Evaluate on {len(dev_dataloader.dataset)} samples")
                self.evaluate(dev_dataloader)
        self.log("Training done")

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        for batch in tqdm.tqdm(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, mask, token_type_ids, start_positions, end_positions = batch
            loss, _, _ = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad(set_to_none=True)

    def evaluate(self, dataloader: DataLoader):
        # trues = []
        # preds = []
        f1_scores = []
        f1_positive_scores = []
        em_scores = []
        em_positive_scores = []
        self.model.eval()
        for _batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, mask, token_type_ids, start_positions, end_positions = batch
            with torch.no_grad():
                _, start_logits, end_logits = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=mask
                )
                start_logits = start_logits.detach().cpu().numpy()
                end_logits = end_logits.detach().cpu().numpy()
                # List[int]
                start_position_preds = start_logits.argmax(axis=-1)
                # List[int]
                end_position_preds = end_logits.argmax(axis=-1)
                # List[List[int]]
                start_positions = start_positions.cpu().numpy()
                # List[List[int]]
                end_positions = end_positions.cpu().numpy()
                for input_id, spos_true, epos_true, spos_pred, epos_pred in zip(input_ids,start_positions, end_positions, start_position_preds, end_position_preds):
                    # spos_true: List[int]
                    # spos_pred: int

                    # List[List[int]]
                    true_indices = []
                    for s, e in zip(spos_true, epos_true):
                        true_indices.append(self.get_indices(s, e))
                    # List[int]
                    pred_indices = self.get_indices(spos_pred, epos_pred)
                    metric = QAMetric(true_indices, pred_indices)
                    f1_scores.append(metric.max_f1_score(only_positive=False))
                    f1_positive_scores.append(
                        metric.max_f1_score(only_positive=True))
                    em_scores.append(
                        metric.max_exact_match(only_positive=False))
                    em_positive_scores.append(
                        metric.max_exact_match(only_positive=True))
                    if metric.max_exact_match(only_positive=True):
                        with open("correct_question_extraction_eval.txt",'a+') as f:
                            f.write(f"{_batch_idx}\n")


        self.log(f"- Average F1-score: {np.nanmean(f1_scores)}")
        self.log(
            f"- Average F1-score POSITIVE: {np.nanmean(f1_positive_scores)}")
        self.log(f"- Average Exact match: {np.nanmean(em_scores)}")
        self.log(
            f"- Average Exact match POSITIVE: {np.nanmean(em_positive_scores)}")

    @staticmethod
    def get_indices(spos: int, epos: int):
        if spos == epos == 0:
            return []
        return list(range(spos, epos+1))

    def convert_word_piece_position(self, word_piece_ids: List[int], spos: int, epos: int):
        """This method does:
            - Converts start_position and end_position to list of indices
            - Merge indices of word-pieces to indices of complete words

            Args:
            - word_piece_ids (List[int]): list of word-piece ids obtained by Bert tokenizer
            - spos (int): start word-piece position of the answer
            - epos (int): end word-piece position of the answer

            Returns:
            - indices (List[int]): indices of complete words of the answer in the context

            Notes:
            Word-piece id at spos and epos are the same and is cls_token_id
            means there is no answer in the context,
            so returns []
        """
        if spos == epos == 0:
            return []
        else:
            word_pieces = self.tokenizer.convert_ids_to_tokens(word_piece_ids)
            for i, e in enumerate(word_pieces):
                if e.startswith("##"):
                    if i < spos:
                        spos -= 1
                    if i < epos:
                        epos -= 1
            return list(range(spos, epos+1))

    def save_model(self, output_dir: Text):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)

    def log(self, content):
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{content}\n")
