import os
import sys
import time
import json
import torch
import GPUtil
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from vocab import Vocab
from model import Transformer
from dataset import read_corpus, NMTDataset
from utils import rate, LabelSmoothing, SimpleLossCompute


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    dataloader,
    model,
    loss_compute,
    optimizer,
    scheduler,
    epoch,
    writer,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
    log_step=40,
    device="cpu",
):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    cuda_item = ["src_ids", "tgt_ids", "src_mask", "tgt_mask", "labels"]
    num_batches = len(dataloader)

    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        for k in cuda_item:
            batch[k] = batch[k].to(device)    

        out = model.forward(
            batch["src_ids"],  batch["tgt_ids"], batch["src_mask"], batch["tgt_mask"],
        )
        loss, loss_node = loss_compute(out, batch["labels"], batch["ntokens"])
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch["src_ids"].shape[0]
            train_state.tokens += batch["ntokens"]

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        
        total_loss += loss
        total_tokens += batch["ntokens"]
        tokens += batch["ntokens"]

        if i % log_step == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss/batch['ntokens']:6.2f} | " + \
                f"Tokens / Sec: {tokens/elapsed:7.1f} | Learning Rate: {lr:6.1e}" 
            )

            global_step = num_batches * epoch + i
            writer.add_scalar(f"Loss/{mode}-step", total_loss / total_tokens, global_step)
            writer.add_scalar(f"lr/{mode}-step", lr, global_step)

            start = time.time()
            tokens = 0

        del loss
        del loss_node

    return total_loss / total_tokens, train_state


def train_worker():
    gpu = 0
    ngpus_per_node = 1
    is_distributed = False
    batch_size = 32
    accum_iter = 10
    num_epochs = 8
    base_lr = 1.0
    warmup = 3000
    model_file_prefix = "zh_en_model_"
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    vocab_path = "../data/vocab.json"
    vocab_data = json.load(open(vocab_path))
    src_word2id = vocab_data["src_word2id"]
    tgt_word2id = vocab_data["tgt_word2id"]
    src_vocab = Vocab(src_word2id)
    tgt_vocab = Vocab(tgt_word2id)

    # 获取训练数据
    train_src_file_path = "../data/zh_en_data/train.zh"
    src_sp_model_path = "../data/src.model"
    train_tgt_file_path  ="../data/zh_en_data/train.en"
    tgt_sp_model_path = "../data/tgt.model"
    train_src_data = read_corpus(train_src_file_path, "src", src_sp_model_path)
    train_tgt_data = read_corpus(train_tgt_file_path, "tgt", tgt_sp_model_path)

    train_dataset = NMTDataset(
        src_data=train_src_data,
        tgt_data=train_tgt_data,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
        # num_workers=8,
        # pin_memory=True,
    )

    # 获取验证数据
    eval_src_file_path = "../data/zh_en_data/dev.zh"
    eval_tgt_file_path = "../data/zh_en_data/dev.en"
    eval_src_data = read_corpus(eval_src_file_path, 'src', src_sp_model_path)
    eval_tgt_data = read_corpus(eval_tgt_file_path, 'tgt', tgt_sp_model_path)

    eval_dataset = NMTDataset(
        src_data=eval_src_data,
        tgt_data=eval_tgt_data,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=eval_dataset.generate_batch,
        # num_workers=8,
        # pin_memory=True,
    )

    pad_idx = tgt_vocab.pad_id

    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    model = Transformer(
        d_model, num_heads, num_layers, num_layers, d_ff, len(src_vocab), len(tgt_vocab), dropout
    )
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0
    
    criterion = LabelSmoothing(
        size=len(tgt_word2id), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=warmup)
    )
    train_state = TrainState()

    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs/{now_time}/"
    os.makedirs(outputs_dir)
    writer = SummaryWriter(outputs_dir)

    for epoch in range(num_epochs):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            eval_dataloader.sampler.set_epoch(epoch)
        
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        train_loss, train_state = run_epoch(
            train_dataloader,
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            epoch=epoch,
            writer=writer,
            mode="train+log",
            accum_iter=accum_iter,
            train_state=train_state,
            device=device,
        )
        writer.add_scalar("Loss/train-epoch", train_loss, epoch)

        GPUtil.showUtilization()
        if is_main_process:
            filename = f"{model_file_prefix}_{epoch}.pth"
            filepath = os.path.join(outputs_dir, filename)
            model.save(filepath)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        with torch.no_grad():
            eval_loss, eval_state = run_epoch(
                eval_dataloader,
                model,
                SimpleLossCompute(module.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
                device=device,
            )
            writer.add_scalar(f"Loss/eval-epoch", eval_loss, epoch)
            torch.cuda.empty_cache()

    if is_main_process:
        filename = f"{model_file_prefix}_final.pth"
        filepath = os.path.join(outputs_dir, filename)
        model.save(filepath)


if __name__ == "__main__":
    os.chdir(sys.path[0])
    train_worker()