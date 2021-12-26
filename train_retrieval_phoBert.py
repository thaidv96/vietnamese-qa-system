from argparse import ArgumentParser
from tabulate import tabulate
from collections import defaultdict
import numpy as np
import pandas as pd
from iteration_utilities import unique_everseen
from tqdm import tqdm
from transformers import get_scheduler
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel
from models.retrieval_model import RetrievalModel, RetrievalDataset
from torch.utils.data import DataLoader
import torch
from python_rdrsegmenter import load_segmenter
segmenter = load_segmenter()


def evaluate(model, epoch, args):
    torch.cuda.empty_cache()
    model.eval()
    passage_embeddings = []
    for doc in tqdm(global_passages):
        passage_inputs = tokenizer(segmenter.tokenize(doc['text']), padding=True,
                                   truncation=True, max_length=256, return_tensors='pt')
        passage_inputs = {k: v.to(device) for k, v in passage_inputs.items()}
        passage_embeddings.append(model.get_representation(
            model.ctx_model, **passage_inputs, fix_model=True).to('cpu').detach().numpy())
    passage_embeddings = np.vstack(passage_embeddings)
    # passage_embeddings = passage_embeddings / \
    #     np.linalg.norm(passage_embeddings, axis=1, keepdims=True)

    result = defaultdict(list)
    for question, passage_id in tqdm(zip(dev_dataset.questions, dev_dataset.context_ids)):
        question_inputs = tokenizer(segmenter.tokenize(question), padding=True,
                                    truncation=True, max_length=50, return_tensors='pt')
        question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
        question_embedding = model.get_representation(
            model.question_model, **question_inputs, fix_model=True).to('cpu').detach().numpy().squeeze()
        global_scores = question_embedding.dot(passage_embeddings.T)
        arg_sort_global_scores = global_scores.argsort()
        for k in [5, 10, 20, 100]:
            global_top_k_indices = arg_sort_global_scores[-k:][::-1]
            result[f'top_{k}_accuracy'].append(
                passage_id in global_passage_df.iloc[global_top_k_indices].id.values)
    eval_str = f"\nRESULT EPOCH: {epoch} \n{tabulate(pd.DataFrame(result).mean().to_frame().T, headers='keys', tablefmt='psql', showindex=False)}"
    print(eval_str)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        required=False,
                        default="vinai/phobert-base")
    parser.add_argument("--training_size", type=int,
                        required=False,
                        default=-1)
    args = parser.parse_args()
    device = torch.device('cuda:2')

    model_name = args.base_model
    batch_size = 4

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    question_model = AutoModel.from_pretrained(model_name)
    ctx_model = AutoModel.from_pretrained(model_name)

    model = RetrievalModel(question_model, ctx_model)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    if args.training_size == -1:
        train_dataset = RetrievalDataset("data/train_retrieval.json")
    else:
        train_dataset = RetrievalDataset("data/train_retrieval.json", sample_size=args.training_size)
    dev_dataset = RetrievalDataset("data/dev_retrieval.json")
    test_dataset = RetrievalDataset("data/test_retrieval.json")
    global_passages = []
    seen_passage_ids = set()
    for dataset in [train_dataset, dev_dataset, test_dataset]:
        for context, passage_id in zip(dataset.contexts, dataset.context_ids):
            if passage_id not in seen_passage_ids:
                global_passages.append({"text": context, "id": passage_id})
                seen_passage_ids.add(passage_id)
    global_passage_df = pd.DataFrame(global_passages)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset)
    test_dataloader = DataLoader(test_dataset)

    num_epochs = 15
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = num_training_steps*0.1
    accum_iter = 8
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model.to(device)

    evaluate(model, 0, args)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        progress_bar = tqdm(train_dataloader, leave=False)
        for batch_num, batch in enumerate(progress_bar):
            batch['question'] = [segmenter.tokenize(
                q) for q in batch['question']]

            question_inputs = tokenizer(batch['question'], padding=True,
                                        truncation=True, max_length=50, return_tensors='pt')
            question_inputs = {k: v.to(device)
                               for k, v in question_inputs.items()}
            context_id = batch['context_id']
            batch['context'] = [segmenter.tokenize(
                c) for c in batch['context']]
            context_inputs = tokenizer([batch['context'][context_id.index(i)] for i in unique_everseen(context_id)], padding=True,
                                       truncation=True, max_length=256, return_tensors='pt')
            context_inputs = {k: v.to(device)
                              for k, v in context_inputs.items()}
            outputs = model(question_inputs, context_inputs, context_id)
            loss = outputs[0]
            loss = loss/accum_iter

            progress_bar.set_description(
                desc=f"Epoch {epoch+1}, Current loss value: {loss.item()}")
            loss.backward()
            if ((batch_num+1) % accum_iter == 0) or (batch_num+1 == len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        print("EVALUATING MODEL...")
        # Get list of global passages:
        evaluate(model, epoch+1, args)

    model.question_model.save_pretrained('checkpoint/question_model')
    model.ctx_model.save_pretrained('checkpoint/ctx_model')
