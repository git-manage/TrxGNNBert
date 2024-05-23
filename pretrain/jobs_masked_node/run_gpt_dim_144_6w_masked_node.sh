#!/bin/bash
cd ../

[ -d data/raw_data ] || mkdir -p data/raw_data
[ -d data/preprocessed ] || mkdir -p data/preprocessed
[ -d data/split ] || mkdir -p data/split

gpt_hidden_dim=144
gpt_num_head=12
tokenizer_dir=$1
token_vocab=$2
token_merge=$3
raw_data_folder='./data/raw_data_6w'
pickle_path='./data/preprocessed/data_6w.pickle'  #./data/preprocessed/data_larger_tokenizaer_6w.pickle
saved_model="saved_model_6w"
# [ -d $saved_model ] || mkdir $saved_model

# export TORCH_USE_CUDA_DSA=1
# CUDA_LAUNCH_BLOCKING=1 
script_name=$(basename "$0")
python main.py \
    --learning_rate 3e-5 \
    --do_train \
    --do_eval \
    --seed 42 \
    --epochs 5 \
    --batch_size 32 \
    --masked_node \
    --device 'cuda:0' \
    --start_step 100 \
    --gradient_accumulation_steps 10 \
    --max_grad_norm 5 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --gnn_hidden_dim 64 \
    --gpt_hidden_dim $gpt_hidden_dim \
    --gpt_num_head $gpt_num_head \
    --mlm_probability 0.15 \
    --sequence 128 \
    --is_tighted_lm_head \
    --output_dir $saved_model \
    --gnn_model_path 'gnn_model.pth' \
    --transformer_model_path 'transformer_model.pth' \
    --emb_model_path 'nn_embedding_model.pth' \
    --raw_data_folder $raw_data_folder \
    --pickle_path $pickle_path \
    --tokenizer_dir $tokenizer_dir \
    --token_vocab $token_vocab \
    --token_merge $token_merge 2>&1 | tee  jobs_masked_node/log_${script_name}_${tokenizer_dir}.txt

    # --train_file './data/split/train.pickle' \
    # --eval_file './data/split/eval.pickle' \
    # --test_file './data/split/test.pickle' \

cd -