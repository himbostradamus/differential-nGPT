problem_name="DiffNGPT_1kctx_10k_lr30e-4"  # Select which configuration to run
runtype="scratch"   # the first run from scratch
#runtype="resume"   # uncomment to resume from the last checkpoint when needed

# Notes: 
# block_size = sequence/context length
# total batch size = gradient_accumulation_steps * batch_size
# if there is a limit on job duration, you will need to implement scratch/resume logic (also check time_limit_seconds and max_iters_per_launch)
# you can adjust max_iters_per_launch to stop training after the specified number of local (within the job) training steps.
# the settings for gradient_accumulation_steps and batch_size are configured for running on 8 nodes (64 GPUs) in parallel.

# Original GPT configurations
if [ "$problem_name" = "GPT_1kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=0 --learning_rate=30e-4 --weight_decay=0.1 --warmup_iters=2000 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=1024 --compile=False --batch_size=8 --gradient_accumulation_steps=64 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=14000"
fi

if [ "$problem_name" = "nGPT_1kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=1 --learning_rate=30e-4 --weight_decay=0.0 --warmup_iters=0 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=1024 --compile=False --batch_size=8 --gradient_accumulation_steps=64 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=14000"
fi

if [ "$problem_name" = "GPT_4kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=0 --learning_rate=30e-4 --weight_decay=0.1 --warmup_iters=2000 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=4096 --compile=False --batch_size=2 --gradient_accumulation_steps=256 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=18000"
fi

if [ "$problem_name" = "nGPT_4kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=1 --learning_rate=30e-4 --weight_decay=0.0 --warmup_iters=0 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=4096 --compile=False --batch_size=2 --gradient_accumulation_steps=256 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=18000"
fi

# New Differential nGPT configurations
if [ "$problem_name" = "DiffNGPT_1kctx_10k_lr30e-4" ]; then
    # Differential nGPT with 1k context - uses nGPT normalization with differential attention
    mycommand=" --init_from='$runtype' --use_nGPT=1 --learning_rate=30e-4 --weight_decay=0.0 --warmup_iters=0 --n_layer=24 --n_head=8 --n_embd=1024 --block_size=1024 --compile=False --batch_size=8 --gradient_accumulation_steps=64 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=14000"
fi

if [ "$problem_name" = "DiffNGPT_4kctx_10k_lr30e-4" ]; then
    # Differential nGPT with 4k context - uses nGPT normalization with differential attention
    mycommand=" --init_from='$runtype' --use_nGPT=1 --learning_rate=30e-4 --weight_decay=0.0 --warmup_iters=0 --n_layer=24 --n_head=8 --n_embd=1024 --block_size=4096 --compile=False --batch_size=2 --gradient_accumulation_steps=256 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=18000"
fi

if [ "$problem_name" = "DiffTransformer_1kctx_10k_lr30e-4" ]; then
    # Standard Transformer with differential attention (no nGPT normalization)
    mycommand=" --init_from='$runtype' --use_nGPT=0 --learning_rate=30e-4 --weight_decay=0.1 --warmup_iters=2000 --n_layer=24 --n_head=8 --n_embd=1024 --block_size=1024 --compile=False --batch_size=8 --gradient_accumulation_steps=64 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=14000"
fi

if [ "$problem_name" = "DiffTransformer_4kctx_10k_lr30e-4" ]; then
    # Standard Transformer with differential attention (no nGPT normalization)
    mycommand=" --init_from='$runtype' --use_nGPT=0 --learning_rate=30e-4 --weight_decay=0.1 --warmup_iters=2000 --n_layer=24 --n_head=8 --n_embd=1024 --block_size=4096 --compile=False --batch_size=2 --gradient_accumulation_steps=256 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=18000"
fi

# Run the command
if [ "$mycommand" != "" ]; then
    torchrun --nnodes 1 --nproc_per_node 8 --rdzv_endpoint=localhost:29501 train.py $mycommand
fi
