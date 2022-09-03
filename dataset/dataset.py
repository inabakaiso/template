from lib2to3.pgen2.tokenize import tokenize
from random import sample
from src.pachage_list import *

def prepare_input(cfg, text, tokenizer, max_len):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class FeedbackDataset(Dataset):
    def __init__(self, cfg, df, max_len, tokenizer):
        self.cfg = cfg
        self.texts = df[cfg.dataset.input_col].values
        self.labels = df[cfg.dataset.target_cols].values
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item], self.tokenizer, self.max_len)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs