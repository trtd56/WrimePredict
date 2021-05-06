import numpy as np
import torch
import torch.nn as nn


loss_fn = nn.BCEWithLogitsLoss()


def train_loop(train_dataloader, model, optimizer, scheduler, device, tqdm):
    losses, lrs = [], []
    model.train()
    optimizer.zero_grad()
    for n_iter, d in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        target = d["target"].to(device)

        output = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        lrs.append(np.array([param_group["lr"] for param_group in optimizer.param_groups]).mean())
        losses.append(loss.item())
    return lrs, losses

def test_loop(test_dataloader, model, device, tqdm):
    losses, predicts = [], []
    model.eval()
    for n_iter, d in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        target = d["target"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, token_type_ids)

        loss = loss_fn(output, target)
        losses.append(loss.item())
        predicts += output.sigmoid().cpu().tolist()

    return predicts, np.array(losses).mean()
