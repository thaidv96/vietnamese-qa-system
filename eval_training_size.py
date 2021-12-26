import os

for training_size in [5000, 10000, 15000]:
    os.system(f"python train_retrieval_phoBert.py --training_size={training_size} > results-vinai_phobert-base.{training_size}.log")