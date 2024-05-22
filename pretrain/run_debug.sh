#!/bin/bash

[ -d saved_model ] || mkdir saved_model
[ -d tokenizer ] || mkdir tokenizer
[ -d data/raw_data ] || mkdir -p data/raw_data
[ -d data/preprocessed ] || mkdir -p data/preprocessed
[ -d data/split ] || mkdir -p data/split

export TORCH_USE_CUDA_DSA=1
CUDA_LAUNCH_BLOCKING=1 python main.py \
    --learning_rate 3e-5 \
    --do_train \
    --do_eval \
    --seed 42 \
    --epochs 5 \
    --batch_size 32 \
    --fp16 \
    --masked_edge \
    --device 'cuda:0' \
    --start_step 100 \
    --gradient_accumulation_steps 10 \
    --max_grad_norm 5 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --gnn_hidden_dim 64 \
    --gpt_hidden_dim 64 \
    --gpt_num_head 8 \
    --mlm_probability 0.15 \
    --sequence 128 \
    --epoch 5 \
    --is_tighted_lm_head \
    --output_dir './saved_model' \
    --gnn_model_path 'gnn_model.pth' \
    --transformer_model_path 'transformer_model.pth' \
    --emb_model_path 'nn_embedding_model.pth' \
    --raw_data_folder './data/raw_data' \
    --pickle_path './data/preprocessed/data.pickle' \
    --tokenizer_dir 'new_tokenizer' \
    --token_vocab 'vocab.json' \
    --token_merge 'merges.txt' \
    --debug
    # --tokenizer_dir 'tokenizer' \
    # --token_vocab 'esperberto-vocab.json' \
    # --token_merge 'esperberto-merges.txt' \
    # --debug
    # --train_file './data/split/train.pickle' \
    # --eval_file './data/split/eval.pickle' \
    # --test_file './data/split/test.pickle' \
