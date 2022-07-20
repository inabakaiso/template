from random import sample
from src.pachage_list import *

class FeedbackDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.text = df['text'].values
        self.target = df['target'].values
        self.tokenizer = tokenizer
        self.discourse_id = df['discourse_id']
        self.essay = df['essay_text']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        
        inputs = self.tokenizer.encode_plus(
                        self.text[item],
                        self.essay[item],
                        truncation=True,
                        add_special_tokens=True,
                        return_token_type_ids = True,
                        max_length=self.cfg.dataset.max_len
                    )
        samples = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.target[item],
        }

        if 'token_type_ids' in inputs:
            samples['token_type_ids'] = inputs['token_type_ids']
        return samples

class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output