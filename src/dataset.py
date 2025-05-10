import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        # 加载 npy 文件
        self.data = np.load(args.data_file).astype("int")
        self.vocab_size = args.vocab_size
        rank_zero_info(f"Loaded data with {self.data.shape[0]} samples, vocab size = {self.vocab_size}")
        self.data_size = len(self.data)
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tokens = self.data[idx]  # tokens shape = (ctx_len,)
        tokens = torch.tensor(tokens, dtype=torch.long)

        input_ids = tokens[:-1]   # 输入是前 ctx_len-1 个 token
        labels = tokens[1:]       # 目标是后 ctx_len-1 个 token
        return input_ids, labels