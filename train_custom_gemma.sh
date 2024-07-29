export CUDA_VISIBLE_DEVICES=0,1,2,3

# 32768

accelerate launch --mixed_precision='bf16' --num_processes 4 --log-dir /home/raidenn/output \
    train.gemma.infini.noclm.py \
    --model_name_or_path='google/gemma-2b' \
    --segment_length=256 \
    --block_size=1024 \
    --dataset_name='wikitext' \
    --dataset_config_name='wikitext-2-raw-v1' \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --weight_decay=1.0 \
    --output_dir='./models/gemma-2b-infini-noclm-wikitext' \
    --checkpointing_steps=10 \
    --num_train_epochs=1 \
    --learning_rate=5e-5 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='tensorboard' \
    --preprocessing_num_workers=4 \
    --with_tracking \
