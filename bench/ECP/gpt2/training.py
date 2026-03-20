import os
import time
import pickle
import hashlib
import argparse
from itertools import islice

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel


# ======================================================
# DATA LOADING
# ======================================================
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
        "Unable to load a streaming Wikipedia dataset. Tried: " + "; ".join(errors)
    )


class GPT2Dataset(Dataset):
    def __init__(self, token_chunks, seq_len, pad_token_id):
        self.token_chunks = token_chunks
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.token_chunks)

    def __getitem__(self, idx):
        ids = self.token_chunks[idx]

        # fixed-length pad/truncate
        if len(ids) < self.seq_len:
            ids = ids + [self.pad_token_id] * (self.seq_len - len(ids))
        else:
            ids = ids[: self.seq_len]

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()

        # causal LM labels: ignore pads
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ======================================================
# TRAIN
# ======================================================
def train(
    epochs=3,
    batch_size=64,
    seq_len=128,
    num_articles=20,
    model_name="gpt2",
    lr=5e-5,
):
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = local_rank == 0

    # Only rank 0 loads and tokenizes; result is broadcast to all ranks.
    # Tokenized chunks are cached to disk to avoid re-streaming on every run.
    if is_main:
        print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    if is_main:
        cache_key = hashlib.md5(f"{model_name}_{num_articles}_{seq_len}".encode()).hexdigest()[:8]
        cache_path = os.path.join(os.path.dirname(__file__), f".token_cache_{cache_key}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading tokenized data from cache ({cache_path})...")
            with open(cache_path, "rb") as f:
                token_chunks = pickle.load(f)
        else:
            print("Streaming Wikipedia and tokenizing...")
            stream = load_wikipedia_stream()
            small_dataset = list(islice(stream, int(num_articles)))
            token_chunks = []
            for row in small_dataset:
                text = row.get("text", "")
                if not text:
                    continue
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    continue
                for i in range(0, len(token_ids), seq_len):
                    chunk = token_ids[i : i + seq_len]
                    if len(chunk) > 0:
                        token_chunks.append(chunk)
            if len(token_chunks) == 0:
                raise RuntimeError("No token chunks were created. Check dataset/tokenizer.")
            with open(cache_path, "wb") as f:
                pickle.dump(token_chunks, f)
            print(f"Tokenized data cached to {cache_path}")
    else:
        token_chunks = None

    obj = [token_chunks]
    dist.broadcast_object_list(obj, src=0)
    token_chunks = obj[0]

    dataset = GPT2Dataset(token_chunks, seq_len=seq_len, pad_token_id=pad_token_id)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
    )

    if is_main:
        print("Building model...")
    config = GPT2Config.from_pretrained(model_name)
    config.pad_token_id = pad_token_id
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model: {model_name}")
        print(f"Model size: {total_params:.2f}M parameters")
        print(f"Using {world_size} GPU(s) with DDP")

    # --- Train loop ---
    model.train()
    for epoch in range(int(epochs)):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        token_count = 0

        torch.cuda.synchronize(device)
        epoch_start = time.time()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()
            optimizer.step()

            total_loss += outputs.loss.item()
            token_count += int(attention_mask.sum().item())

        torch.cuda.synchronize(device)
        epoch_time = time.time() - epoch_start

        # Aggregate token_count across all ranks for total throughput
        token_count_tensor = torch.tensor(token_count, dtype=torch.long, device=device)
        dist.all_reduce(token_count_tensor, op=dist.ReduceOp.SUM)

        if is_main:
            avg_loss = total_loss / max(len(dataloader), 1)
            tokens_per_sec = token_count_tensor.item() / max(epoch_time, 1e-12)
            print(
                f"Epoch {epoch + 1} done. "
                f"Avg Loss {avg_loss:.4f} | "
                f"Train time {epoch_time:.2f}s | "
                f"Tokens/sec {tokens_per_sec:.2f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-articles", type=int, default=20)
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_articles=args.num_articles,
        model_name=args.model_name,
        lr=args.lr,
    )


# import os
# import time
# import pickle
# import hashlib
# import argparse
# from itertools import islice

# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader, DistributedSampler

# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


# # ======================================================
# # DATA LOADING
# # ======================================================
# def load_wikipedia_stream():
#     """Load a streaming Wikipedia split with compatibility across datasets versions."""
#     candidates = [
#         ("wikimedia/wikipedia", "20231101.en"),
#         ("wikimedia/wikipedia", "20220301.en"),
#         ("wikipedia", "20220301.en"),
#     ]
#     errors = []
#     for name, config in candidates:
#         try:
#             return load_dataset(name, config, split="train", streaming=True)
#         except Exception as exc:
#             errors.append(f"{name}/{config}: {exc}")
#     raise RuntimeError(
#         "Unable to load a streaming Wikipedia dataset. Tried: " + "; ".join(errors)
#     )


# class CausalLMDataset(Dataset):
#     def __init__(self, token_chunks, seq_len, pad_token_id):
#         self.token_chunks = token_chunks
#         self.seq_len = seq_len
#         self.pad_token_id = pad_token_id

#     def __len__(self):
#         return len(self.token_chunks)

#     def __getitem__(self, idx):
#         ids = self.token_chunks[idx]

#         if len(ids) < self.seq_len:
#             ids = ids + [self.pad_token_id] * (self.seq_len - len(ids))
#         else:
#             ids = ids[: self.seq_len]

#         input_ids = torch.tensor(ids, dtype=torch.long)
#         attention_mask = (input_ids != self.pad_token_id).long()

#         labels = input_ids.clone()
#         labels[attention_mask == 0] = -100

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels,
#         }


# # ======================================================
# # TRAIN
# # ======================================================
# def train(
#     epochs=3,
#     batch_size=4,
#     seq_len=128,
#     num_articles=20,
#     model_name="EleutherAI/gpt-j-6B",
#     lr=2e-5,
# ):
#     dist.init_process_group("nccl")
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = dist.get_world_size()
#     device = torch.device(f"cuda:{local_rank}")
#     torch.cuda.set_device(device)
#     is_main = local_rank == 0

#     if is_main:
#         print("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     pad_token_id = tokenizer.pad_token_id

#     if is_main:
#         cache_key = hashlib.md5(f"{model_name}_{num_articles}_{seq_len}".encode()).hexdigest()[:8]
#         cache_path = os.path.join(os.path.dirname(__file__), f".token_cache_{cache_key}.pkl")

#         if os.path.exists(cache_path):
#             print(f"Loading tokenized data from cache ({cache_path})...")
#             with open(cache_path, "rb") as f:
#                 token_chunks = pickle.load(f)
#         else:
#             print("Streaming Wikipedia and tokenizing...")
#             stream = load_wikipedia_stream()
#             small_dataset = list(islice(stream, int(num_articles)))
#             token_chunks = []

#             for row in small_dataset:
#                 text = row.get("text", "")
#                 if not text:
#                     continue
#                 token_ids = tokenizer.encode(text, add_special_tokens=False)
#                 if not token_ids:
#                     continue
#                 for i in range(0, len(token_ids), seq_len):
#                     chunk = token_ids[i : i + seq_len]
#                     if len(chunk) > 0:
#                         token_chunks.append(chunk)

#             if len(token_chunks) == 0:
#                 raise RuntimeError("No token chunks were created. Check dataset/tokenizer.")

#             with open(cache_path, "wb") as f:
#                 pickle.dump(token_chunks, f)
#             print(f"Tokenized data cached to {cache_path}")
#     else:
#         token_chunks = None

#     obj = [token_chunks]
#     dist.broadcast_object_list(obj, src=0)
#     token_chunks = obj[0]

#     dataset = CausalLMDataset(token_chunks, seq_len=seq_len, pad_token_id=pad_token_id)
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=4,
#         pin_memory=True,
#     )

#     if is_main:
#         print("Building model...")

#     config = AutoConfig.from_pretrained(model_name)
#     config.pad_token_id = pad_token_id

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         config=config,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#     ).to(device)

#     model = DDP(model, device_ids=[local_rank], output_device=local_rank)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#     if is_main:
#         total_params = sum(p.numel() for p in model.parameters()) / 1e9
#         print(f"Model: {model_name}")
#         print(f"Model size: {total_params:.2f}B parameters")
#         print(f"Using {world_size} GPU(s) with DDP")
#         print("Training dtype: bfloat16")

#     model.train()
#     for epoch in range(int(epochs)):
#         sampler.set_epoch(epoch)
#         total_loss = 0.0
#         token_count = 0

#         torch.cuda.synchronize(device)
#         epoch_start = time.time()

#         for batch in dataloader:
#             input_ids = batch["input_ids"].to(device, non_blocking=True)
#             attention_mask = batch["attention_mask"].to(device, non_blocking=True)
#             labels = batch["labels"].to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#                 outputs = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels,
#                 )
#                 loss = outputs.loss

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             token_count += int(attention_mask.sum().item())

#         torch.cuda.synchronize(device)
#         epoch_time = time.time() - epoch_start

#         token_count_tensor = torch.tensor(token_count, dtype=torch.long, device=device)
#         dist.all_reduce(token_count_tensor, op=dist.ReduceOp.SUM)

#         loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
#         dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

#         if is_main:
#             avg_loss = loss_tensor.item() / max(len(dataloader) * world_size, 1)
#             tokens_per_sec = token_count_tensor.item() / max(epoch_time, 1e-12)
#             print(
#                 f"Epoch {epoch + 1} done. "
#                 f"Avg Loss {avg_loss:.4f} | "
#                 f"Train time {epoch_time:.2f}s | "
#                 f"Tokens/sec {tokens_per_sec:.2f}"
#             )

#     dist.destroy_process_group()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--batch-size", type=int, default=1)
#     parser.add_argument("--seq-len", type=int, default=128)
#     parser.add_argument("--num-articles", type=int, default=20)
#     parser.add_argument("--model-name", type=str, default="EleutherAI/gpt-j-6B")
#     parser.add_argument("--lr", type=float, default=2e-5)
#     args = parser.parse_args()

#     train(
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         seq_len=args.seq_len,
#         num_articles=args.num_articles,
#         model_name=args.model_name,
#         lr=args.lr,
#     )