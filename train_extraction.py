from models.extraction_model import BertForQuestionAnswering, RobertaForQuestionAnswering, Trainer
from data_helper.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, PhobertTokenizer
import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_type", type=str, default='bert')
parser.add_argument("--device", type=str, default='cuda:1')
parser.add_argument("--negative_sampling", type=str, default='true')
parser.add_argument("--num_epochs", type=int, default=5)
args = parser.parse_args()

if args.model_type == 'bert':
    if args.negative_sampling=='true':
        train_ids_path = 'data/train_bert_extraction_ids.txt'
        dev_ids_path =  'data/dev_bert_extraction_ids.txt'
        log_file = 'logs/reader_logs/bert.log'
        model_dir = 'checkpoint/bert_extraction_model'

    else:
        train_ids_path = 'data/train_bert_extraction_ids_no_negative.txt'
        dev_ids_path =  'data/dev_bert_extraction_ids_no_negative.txt'
        log_file = 'logs/reader_logs/bert_no_negative.log'
        model_dir = 'checkpoint/bert_extraction_model_no_negative'
    
    pretrained_model_name = 'bert'
    pretrained_model_path ='bert-base-multilingual-cased'
else:
    if args.negative_sampling=='true':
        train_ids_path = 'data/train_phobert_extraction_ids.txt'
        dev_ids_path =  'data/dev_phobert_extraction_ids.txt'
        log_file = 'logs/reader_logs/phobert.log'
        model_dir = 'checkpoint/phobert_extraction_model'
    
    else:
        train_ids_path = 'data/train_phobert_extraction_ids_no_negative.txt'
        dev_ids_path =  'data/test_phobert_extraction_ids_no_negative.txt'
        log_file = 'logs/reader_logs/phobert_no_negative.log'
        model_dir = 'checkpoint/phobert_extraction_model_no_negative'
    pretrained_model_name = 'phobert'
    pretrained_model_path = 'vinai/phobert-base'

maxlen = 256
batch_size = 32
n_epochs = args.num_epochs
# if args.training_size > -1:
#     n_epochs = 10 - args.training_size//5000
device = args.device

train_instances = open(train_ids_path, "r",
                       encoding="utf-8").read().splitlines()
dev_instances = open(dev_ids_path, "r", encoding="utf-8").read().splitlines()

if pretrained_model_name == "phobert":
    tokenizer = PhobertTokenizer.from_pretrained(
        pretrained_model_path, do_lower_case=False, remove_accents=False)
    model = RobertaForQuestionAnswering.from_pretrained(
        pretrained_model_path, num_labels=2)
else:
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_path, do_lower_case=False, remove_accents=False)
    model = BertForQuestionAnswering.from_pretrained(
        pretrained_model_path, num_labels=2)

train_dataset = CustomDataset(
    train_instances, maxlen=maxlen, pad_token_id=tokenizer.pad_token_id)
dev_dataset = CustomDataset(
    dev_instances, maxlen=maxlen, pad_token_id=tokenizer.pad_token_id)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

trainer = Trainer(pretrained_model=model, tokenizer=tokenizer, device=device)
trainer.train(train_dataloader, dev_dataloader, n_epochs=n_epochs,
              model_dir=model_dir, log_file=log_file)
