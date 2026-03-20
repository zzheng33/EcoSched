import os
import math
import random
import time
import pickle
import hashlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from datasets import load_dataset
from transformers import BertTokenizer
from itertools import islice

# ======================================================
# BERT CONFIG
# ======================================================
class BERTConfig:
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 1024
        self.num_hidden_layers = 24
        self.num_attention_heads = 16
        self.intermediate_size = 4096
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.layer_norm_eps = 1e-12

# ======================================================
# BERT MODEL COMPONENTS (same as before)
# ======================================================
class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        word_emb = self.word_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        embeddings = word_emb + position_emb + token_type_emb
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        return context.view(*new_shape)

class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)

class BERTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        return self.output(self_output, hidden_states)

class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))

class BERTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)

class BERTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)
    def forward(self, hidden_states, attention_mask=None):
        attn_out = self.attention(hidden_states, attention_mask)
        inter_out = self.intermediate(attn_out)
        return self.output(inter_out, attn_out)

class BERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class BERTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        first_token = hidden_states[:, 0]
        return self.activation(self.dense(first_token))

class BERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        model_dtype = self.embeddings.word_embeddings.weight.dtype
        ext_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=model_dtype)
        ext_mask = (1.0 - ext_mask) * -10000.0
        emb = self.embeddings(input_ids, token_type_ids)
        enc = self.encoder(emb, ext_mask)
        pooled = self.pooler(enc)
        return enc, pooled

class BERTLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    def forward(self, hidden_states):
        x = F.gelu(self.transform(hidden_states))
        x = self.LayerNorm(x)
        return self.decoder(x) + self.bias

class BERTPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BERTLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    def forward(self, seq_out, pooled_out):
        return self.predictions(seq_out), self.seq_relationship(pooled_out)

class BERTForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        self.cls = BERTPreTrainingHeads(config)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, next_sentence_label=None):
        seq_out, pooled_out = self.bert(input_ids, attention_mask, token_type_ids)
        pred_scores, seq_rel = self.cls(seq_out, pooled_out)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(pred_scores.view(-1, self.bert.embeddings.word_embeddings.num_embeddings),
                                masked_lm_labels.view(-1))
            nsp_loss = loss_fct(seq_rel.view(-1, 2), next_sentence_label.view(-1))
            return mlm_loss + nsp_loss, mlm_loss, nsp_loss
        return pred_scores, seq_rel

# ======================================================
# DATASET
# ======================================================
class BERTDataset(Dataset):
    def __init__(self, sentences, max_len=128, vocab_size=30522):
        self.sentences = sentences
        self.max_len = max_len
        self.vocab_size = vocab_size

    def create_masked_lm_predictions(self, tokens):
        output_tokens = tokens.clone()
        labels = torch.full_like(tokens, -100)
        for i in range(1, len(tokens) - 1):
            if random.random() < 0.15:
                labels[i] = tokens[i]
                if random.random() < 0.8:
                    output_tokens[i] = 103
                elif random.random() < 0.5:
                    output_tokens[i] = random.randint(1, self.vocab_size - 1)
        return output_tokens, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens_a = self.sentences[idx]
        if random.random() < 0.5:
            is_next = 1
            tokens_b = self.sentences[(idx + 1) % len(self.sentences)]
        else:
            is_next = 0
            tokens_b = self.sentences[random.randint(0, len(self.sentences) - 1)]
        tokens_a = tokens_a.tolist()
        tokens_b = tokens_b.tolist()
        tokens = [101] + tokens_a[:self.max_len // 2 - 2] + [102] + \
                 tokens_b[:self.max_len // 2 - 1] + [102]
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        tokens = torch.tensor(tokens, dtype=torch.long)
        seg_len = min(len(tokens_a) + 2, self.max_len // 2)
        token_type_ids = torch.cat([
            torch.zeros(seg_len, dtype=torch.long),
            torch.ones(self.max_len - seg_len, dtype=torch.long)
        ])
        attention_mask = (tokens != 0).long()
        masked_tokens, masked_labels = self.create_masked_lm_predictions(tokens)
        return {
            'input_ids': masked_tokens,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'masked_lm_labels': masked_labels,
            'next_sentence_label': torch.tensor(is_next, dtype=torch.long)
        }

def load_wikipedia_stream():
    """Load a streaming Wikipedia split with compatibility across datasets versions."""
    candidates = [
        ("wikimedia/wikipedia", "20231101.en"),
        ("wikimedia/wikipedia", "20220301.en"),
        ("wikipedia", "20220301.en"),
    ]
    errors = []
    for name, config in candidates:
        try:
            return load_dataset(name, config, split="train", streaming=True)
        except Exception as exc:
            errors.append(f"{name}/{config}: {exc}")
    raise RuntimeError(
        "Unable to load a streaming Wikipedia dataset. Tried: "
        + "; ".join(errors)
    )


# ======================================================
# TRAIN
# ======================================================
def train(epochs=3, batch_size=4):
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = local_rank == 0

    # Only rank 0 loads and tokenizes; result is broadcast to all ranks.
    # Tokenized sentences are cached to disk to avoid re-streaming on every run.
    if is_main:
        cache_path = os.path.join(os.path.dirname(__file__), ".token_cache_bert.pkl")
        if os.path.exists(cache_path):
            print(f"Loading tokenized data from cache ({cache_path})...")
            with open(cache_path, "rb") as f:
                tokenized_sentences = pickle.load(f)
        else:
            print("Streaming Wikipedia and tokenizing...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            stream = load_wikipedia_stream()
            small_dataset = list(islice(stream, 10))
            tokenized_sentences = []
            for row in small_dataset:
                tokens = tokenizer.encode(row["text"], add_special_tokens=False)
                if len(tokens) > 0:
                    for i in range(0, len(tokens), 128):
                        chunk = tokens[i:i+128]
                        tokenized_sentences.append(torch.tensor(chunk))
            with open(cache_path, "wb") as f:
                pickle.dump(tokenized_sentences, f)
            print(f"Tokenized data cached to {cache_path}")
    else:
        tokenized_sentences = None

    obj = [tokenized_sentences]
    dist.broadcast_object_list(obj, src=0)
    tokenized_sentences = obj[0]

    dataset = BERTDataset(tokenized_sentences, max_len=128)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    config = BERTConfig()
    model = BERTForPreTraining(config).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model size: {total_params:.2f}M parameters")
        print(f"Using {world_size} GPU(s) with DDP")

    model.train()
    for epoch in range(int(epochs)):
        sampler.set_epoch(epoch)
        total_loss = 0
        token_count = 0

        torch.cuda.synchronize(device)
        epoch_train_start = time.time()

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            masked_lm_labels = batch['masked_lm_labels'].to(device, non_blocking=True)
            next_sentence_label = batch['next_sentence_label'].to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, mlm_loss, nsp_loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                masked_lm_labels=masked_lm_labels,
                next_sentence_label=next_sentence_label
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            token_count += int(attention_mask.sum().item())

        torch.cuda.synchronize(device)
        epoch_train_time = time.time() - epoch_train_start

        token_count_tensor = torch.tensor(token_count, dtype=torch.long, device=device)
        dist.all_reduce(token_count_tensor, op=dist.ReduceOp.SUM)

        if is_main:
            tokens_per_sec = token_count_tensor.item() / max(epoch_train_time, 1e-12)
            avg_loss = total_loss / len(dataloader)
            print(
                f"Epoch {epoch+1} done. Avg Loss {avg_loss:.4f} | "
                f"Train time {epoch_train_time:.2f}s | Tokens/sec {tokens_per_sec:.2f}"
            )

    dist.destroy_process_group()

# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
