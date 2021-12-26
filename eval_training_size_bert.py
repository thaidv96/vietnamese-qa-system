import os

for training_size in [5000, 10000, 15000, -1]:
    os.system(f"python train_retrieval.py --training_size={training_size} > results-bert-base.{training_size}.log")