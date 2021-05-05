import torch

class WrimeDataset:
    def __init__(self, toks, targets):
        self.toks = toks
        self.targets = targets

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, item):
        tok = self.toks[item]
        target = self.targets[item]

        input_ids = torch.tensor(tok["input_ids"])
        attention_mask = torch.tensor(tok["attention_mask"])
        token_type_ids = torch.tensor(tok["token_type_ids"])
        target = torch.tensor(target).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "target": target,
        }
