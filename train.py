import os
import time
import math
import pickle
import sys
import argparse
from contextlib import nullcontext
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import functional as F
from datetime import timedelta

from model import DifferentialNGPT, GPTConfig


def get_args():
    parser = argparse.ArgumentParser(description='Differential nGPT Training Script')
    
    # Model configuration
    parser.add_argument('--n_layer', type=int, default=12, help='number of layers')
    parser.add_argument('--n_head', type=int, default=12, help='number of attention heads')
    parser.add_argument('--n_embd', type=int, default=1024, help='embedding dimension')
    parser.add_argument('--block_size', type=int, default=1024, help='context length')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--bias', action='store_true', help='use bias in layers')
    parser.add_argument('--use_ngpt', type=int, default=1, help='use nGPT normalization (1) or not (0)')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8, help='batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64, help='gradient accumulation steps')
    parser.add_argument('--max_iters', type=int, default=600000, help='maximum training iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='learning rate decay iterations')
    parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
    parser.add_argument('--log_interval', type=int, default=10, help='logging interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='evaluation iterations')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping value')
    
    # Learning rate and optimizer
    parser.add_argument('--learning_rate', type=float, default=15e-4, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0, help='minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=0, help='warmup iterations')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.95, help='beta2 for Adam')
    parser.add_argument('--decay_lr', action='store_true', help='decay learning rate')
    
    # Distributed training
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype to use')
    parser.add_argument('--compile', action='store_true', help='use torch.compile')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
    
    # I/O
    parser.add_argument('--dataset', type=str, default='openwebtext', help='dataset name')
    parser.add_argument('--out_dir', type=str, default='./', help='output directory')
    parser.add_argument('--init_from', type=str, default='scratch', help='initialize from scratch/resume/gpt2*')
    parser.add_argument('--eval_only', action='store_true', help='only run evaluation')
    parser.add_argument('--wandb_log', action='store_true', help='log to wandb')
    parser.add_argument('--wandb_project', type=str, default='differential-ngpt', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='differential-ngpt', help='wandb run name')
    
    # Time limits
    parser.add_argument('--time_limit_seconds', type=int, default=1000000000, help='stop after x seconds')
    parser.add_argument('--max_iters_per_launch', type=int, default=1000000000, help='max iterations per launch')
    
    args = parser.parse_args()
    
    # Configure nGPT specific hyperparameters based on use_ngpt flag
    if args.use_ngpt:
        if not args.warmup_iters:
            args.warmup_iters = 0
        if args.weight_decay == 0.1:  # If not set explicitly, use default for nGPT
            args.weight_decay = 0.0
        args.base_scale = 1.0 / args.n_embd ** 0.5
    else:
        if not args.warmup_iters:
            args.warmup_iters = 2000
        if args.weight_decay == 0.0:  # If not set explicitly, use default for standard Transformer
            args.weight_decay = 0.1
        args.base_scale = 0.02
    
    return args


def get_batch(data_dir, split, block_size, batch_size, device_type):
    """Load a batch of data from disk."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Random offsets
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Load samples and their targets
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Pin memory for faster GPU transfer
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, eval_iters, block_size, batch_size, device_type, ctx):
    """Estimate the loss on train and validation splits."""
    model.eval()
    out = {}
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, split, block_size, batch_size, device_type)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


def get_lr(args, iter_num):
    """Get learning rate based on schedule."""
    # Linear warmup
    if iter_num < args.warmup_iters:
        return args.learning_rate * iter_num / args.warmup_iters
    
    # After decay iterations, return min learning rate
    if iter_num > args.lr_decay_iters:
        return args.min_lr
    
    # Cosine decay
    decay_ratio = (iter_num - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def log_parameter_stats(model, args):
    """Log interesting statistics about model parameters."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        transformer = model.module.transformer
        config = model.module.config
        module = model.module
    else:
        transformer = model.transformer
        config = model.config
        module = model
    
    if args.use_ngpt:
        resstr = "%.5f " % torch.mean(module.sz * (module.sz_init_value/module.sz_init_scaling))
        
        for layer_idx in range(0, config.n_layer):
            block = transformer["h"][layer_idx] 
            sqk = block.sqk * (block.sqk_init_value/block.sqk_init_scaling)
            attn_alpha = block.attn_alpha * (block.attn_alpha_init_value / block.attn_alpha_init_scaling)
            mlp_alpha = block.mlp_alpha * (block.mlp_alpha_init_value / block.mlp_alpha_init_scaling)
            suv = block.suv * (block.suv_init_value/block.suv_init_scaling)

            resstr = resstr + "%.5f " % torch.mean(sqk)
            resstr = resstr + "%.5f " % torch.mean(attn_alpha)
            resstr = resstr + "%.5f " % torch.mean(mlp_alpha)
            resstr = resstr + "%.5f " % torch.mean(suv)
            
        # Add differential attention lambda stats
        for layer_idx in range(0, config.n_layer):
            block = transformer["h"][layer_idx]
            lambda_val = torch.exp(torch.sum(block.lambda_q1 * block.lambda_k1, dim=-1).float()) - \
                          torch.exp(torch.sum(block.lambda_q2 * block.lambda_k2, dim=-1).float()) + \
                          block.lambda_init_value
            resstr = resstr + "%.5f " % torch.mean(lambda_val)
         
        return resstr
    else:
        return ""


def main():
    # Get arguments
    args = get_args()
    
    # Set up distributed training
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend=args.backend,
            timeout=timedelta(milliseconds=20*60000)  # 20-minute timeout
        )  
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        # Adjust gradient accumulation for distributed training
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
        dist.barrier()
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        
    # Calculate tokens per iteration for logging
    tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # Create output directory
    if master_process:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    # Set seed for reproducibility
    local_seed = seed_offset
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)

    # Set device and precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    device = torch.device(args.device)
    
    # Set precision
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load data
    tdataloading_begin = time.time()
    if os.path.exists('./../../data'):
        data_dir = os.path.join('./../../data', args.dataset)
    else:   
        data_dir = os.path.join('data', args.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    print("Data loading time: %f sec" % (time.time()-tdataloading_begin))

    # Attempt to get vocabulary size from metadata
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # Initialize model
    iter_num = 0
    tmodelinit_begin = time.time()
    
    # Configure model
    model_args = dict(
        use_nGPT=args.use_ngpt, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd, 
        block_size=args.block_size, 
        base_scale=args.base_scale,
        bias=args.bias, 
        vocab_size=None, 
        dropout=args.dropout
    )
    
    if args.init_from == 'scratch':
        # Initialize from scratch
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = DifferentialNGPT(gptconf)
    elif args.init_from == 'resume':
        # Resume from checkpoint
        print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # Force certain config attributes to match for resuming
        for k in ['use_nGPT', 'base_scale', 'n_layer', 'n_head',  'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # Create model
        gptconf = GPTConfig(**model_args)
        model = DifferentialNGPT(gptconf)
        state_dict = checkpoint['model']
        # Fix state dict keys if needed
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
    
    # Move model to device
    model.to(device)
    print("Model initialization/loading time: %f sec" % (time.time()-tmodelinit_begin))

    # Set up optimizer
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type
    )
    
    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # Free up memory

    # Compile model if requested
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # Wrap model in DDP if needed
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Set up wandb logging
    if args.wandb_log and master_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Start training
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model  # Unwrap DDP container if needed

    # Print hyperparameters
    if master_process:
        print("learning_rate:", args.learning_rate)
        print("min_lr:", args.min_lr)
        print("max_iters:", args.max_iters)
        print("lr_decay_iters:", args.lr_decay_iters)
        print("warmup_iters:", args.warmup_iters)
        print("batch_size:", args.batch_size)
        print("gradient_accumulation_steps:", args.gradient_accumulation_steps)
        print("block_size:", args.block_size)
        print("weight_decay:", args.weight_decay)

    # Setup for logging
    stat_fname = os.path.join(args.out_dir, "stat")
    if master_process:
        if args.init_from == 'scratch':
            file = open(stat_fname, "w")
            resstr = f"{0:.6e} {0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0:.4e} {0.0:.4e}"
            resstr = resstr + log_parameter_stats(model, args) + "\n"
            file.write(resstr)
            arguments = sys.argv
            fname_arg = os.path.join(args.out_dir, "args")
            with open(fname_arg, 'w') as file_arg:
                for arg in arguments:
                    file_arg.write(arg + '\n')
        if args.init_from == 'resume':
            file = open(stat_fname, "a")

    # Log time spent
    time_spent = time.time() - t0
    print(f"Time spent: {time_spent} seconds")
    starting_iter_num = iter_num
    print("starting_iter_num:", iter_num)

    # Handle nGPT normalization
    if args.use_ngpt == 1:
        raw_model.normalize_matrices()

    # Get first batch
    X, Y = get_batch(data_dir, 'train', args.block_size, args.batch_size, device_type)

    # Training loop
    while True:
        if local_iter_num > args.max_iters_per_launch:
            break

        # Use different seed for each iteration to ensure randomness
        local_seed = 100 * iter_num + seed_offset
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed(local_seed)

        # Set learning rate
        lr = get_lr(args, iter_num) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and checkpointing
        if iter_num % args.eval_interval == 0 and master_process:
            rng_state_pytorch = torch.get_rng_state()
            rng_state_bytes = rng_state_pytorch.numpy().tobytes()
            losses = estimate_loss(model, data_dir, args.eval_iters, args.block_size, args.batch_size, device_type, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
           
            if args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr
                })

            # Save checkpoint
            if iter_num > starting_iter_num:
                tcheckpointsaving_begin = time.time()
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'config': vars(args),
                    'rng_state_pytorch_bytes': rng_state_bytes,
                    'rng_state_numpy': np.random.get_state()
                }
                print(f"saving checkpoint to {args.out_dir}")
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
                print("Checkpoint saving time: %f sec" % (time.time()-tcheckpointsaving_begin))
        
        # Exit if evaluation only
        if iter_num == 0 and args.eval_only:
            break

        # Forward and backward passes with gradient accumulation
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                # Only sync gradients on the last micro step
                model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
            
            # Forward pass
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps  # Scale loss
            
            # Prefetch next batch
            X, Y = get_batch(data_dir, 'train', args.block_size, args.batch_size, device_type)
            
            # Backward pass
            loss.backward()

        # Gradient clipping
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            lossf = loss.item() * args.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.6f}, time {dt*1000:.2f}ms")
        
        # Apply nGPT normalization
        if args.use_ngpt == 1:
            raw_model.normalize_matrices()

        # Periodic learning rate print
        if (iter_num % 100 == 0) and master_process:
            print(f"lr={lr}")

        # Log to stats file
        if master_process:
            resstr = f"{iter_num:.6e} {lr:.4e} {losses['train']:.4e} {losses['val']:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0.0:.4e} {0:.4e} {0.0:.4e} "
            resstr = resstr + log_parameter_stats(model, args) + "\n"
            
            file.write(resstr)
            file.flush()

            # Mark as finished if max iterations reached
            if iter_num >= args.max_iters:
                finished_fname = os.path.join(args.out_dir, "finished")
                finished_file = open(finished_fname, "w")
                finished_file.write("1")
                finished_file.close()

        # Check time limit
        if time.time() - t0 > args.time_limit_seconds:
            break

        # Increment iteration counters
        iter_num += 1
        local_iter_num += 1
        
        # Exit if max iterations reached
        if iter_num > args.max_iters:
            break
            
    # Log total time spent
    time_spent = time.time() - t0
    print(f"Time spent: {time_spent} seconds")
    
    # Clean up distributed training
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
