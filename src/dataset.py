import numpy as np
import torch
from torch.utils.data import Dataset

ID_DOT  = 0
ID_LPAR = 1
ID_RPAR = 2
ID_A    = 3
ID_C    = 4
ID_G    = 5
ID_U    = 6
ID_NL   = 7
ID_PAD  = 8

NUC_SET = {ID_A, ID_C, ID_G, ID_U}

class MyDataset(Dataset):

    def __init__(self, args, use_mmap: bool = True, seq_only: bool = True):

        self.data = np.load(args.data_file, mmap_mode='r' if use_mmap else None)
        self.vocab_size = args.vocab_size
        self.seq_only = seq_only
        self.data_size = len(self.data)
    def __len__(self):
        return self.data.shape[0]

    def _effective_tokens(self, row: np.ndarray) -> torch.Tensor:

        nz = np.nonzero(row != ID_PAD)[0]
        T  = int(nz[-1]) + 1 if nz.size > 0 else 2
        T  = max(T, 2)
  
        return torch.from_numpy(row[:T].astype(np.int64, copy=False))

    def __getitem__(self, idx):
        tokens = self._effective_tokens(self.data[idx])  # [T_eff]

        input_ids = tokens[:-1]                          # [T-1]
        labels    = tokens[1:]                           # [T-1]


        if self.seq_only:
     
            # tokens: [S ... NL  X ...]
            nl_pos = (tokens == ID_NL).nonzero(as_tuple=True)[0]
            if nl_pos.numel() > 0:
                p = nl_pos[0].item()         
          
                mask = torch.zeros_like(labels, dtype=torch.float32)
                start = min(p + 1, labels.numel())
                if start < labels.numel():
                    mask[start:] = 1.0
               
                if mask.sum() == 0:
                    mask = ((labels == ID_A) | (labels == ID_C) | (labels == ID_G) | (labels == ID_U)).float()
            else:
          
                mask = ((labels == ID_A) | (labels == ID_C) | (labels == ID_G) | (labels == ID_U)).float()
        else:
 
            mask = ((labels == ID_A) | (labels == ID_C) | (labels == ID_G) | (labels == ID_U)).float()

        return input_ids, labels, mask
    
