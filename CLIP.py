import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

TSCOLS = 256
TRAIN_DATA = "train_val_data/BTC_train.csv"

class CFG:
    debug = False
    batch_size = 32
    num_workers = 0
    head_lr = 1e-3
    timeseries_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model_name = 'resnet50'
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    timeseries_embedding = 256
    max_length = 200

    pretrained = True # for  text encoder
    trainable = True # for text encoder
    temperature = 1.0

    # image size
    # size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, timeseries, captions, tokenizer):
        self.timeseries = list(timeseries)
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.normalizer = 'none'

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        if self.normalizer =='none':
            item['timeseries'] = torch.tensor(self.timeseries[idx]).float()
        if self.normalizer =='minmax':
            item['timeseries'] = (torch.tensor(self.timeseries[idx]).float() - torch.tensor(self.timeseries[idx]).float().min()) / (torch.tensor(self.timeseries[idx]).float().max() - torch.tensor(self.timeseries[idx]).float().min())
        if self.normalizer =='meansd':
            item['timeseries'] = (torch.tensor(self.timeseries[idx]).float() - torch.tensor(self.timeseries[idx]).float().mean()) / torch.tensor(self.timeseries[idx]).float().std()
            
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel(config=AutoConfig())
            
        for p in self.model.parameters():
            p.requires_grad = False

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        text_embedding=CFG.text_embedding,
        timeseries_embedding=CFG.timeseries_embedding,
        use_timeseries_projection_head=True,
    ):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.ts_projection = ProjectionHead(embedding_dim=timeseries_embedding)
        self.temperature = temperature
        self.use_timeseries_projection_head = use_timeseries_projection_head

    def forward(self, batch):
        timeseries_features = batch["timeseries"]
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        if self.use_timeseries_projection_head:
            timeseries_embeddings = self.ts_projection(timeseries_features)
        else:
            timeseries_embeddings = timeseries_features
            
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ timeseries_embeddings.T) / self.temperature
        timeseries_similarity = timeseries_embeddings @ timeseries_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (timeseries_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
def make_train_valid_dfs():
    dataframe = pd.read_csv(TRAIN_DATA, lineterminator='\n')
    max_id = dataframe.shape[0]
    timeseries_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        timeseries_ids, size=int(0.2 * len(timeseries_ids)), replace=False
    )
    train_ids = [id_ for id_ in timeseries_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe.index.isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe.index.isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    dataset = CLIPDataset(
        dataframe.iloc[:, -TSCOLS:].values,
        dataframe.iloc[:, 0].values,
        tokenizer=tokenizer,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

    
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["timeseries"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["timeseries"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def get_ts_embeddings(valid_df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel()
    model.load_state_dict(torch.load(model_path) if CFG.device == "cuda" else torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    valid_timeseries_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            valid_timeseries_embeddings.append(batch["timeseries"])
    return model, torch.cat(valid_timeseries_embeddings)

def find_matches(model, timeseries_embeddings, query, n=9):
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    timeseries_embeddings_n = F.normalize(timeseries_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ timeseries_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches =  indices[::5]

    return matches

def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": {}, "lr": CFG.timeseries_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain({}, model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)
        

