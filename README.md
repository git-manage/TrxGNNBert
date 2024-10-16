# Pretrain TrxGNNBert Model

## Data
```
Options
--raw_data_folder : folder path to the *.pl files, `raw_data` for all all, `raw_data_6w` for 6w trx.
--pickle_path : save preprocessed graph daa
```

## Tokenizer
`tokenizer` is a small tokenizer with about 5,000 tokens
`tokenizer_larger` is a large tokenzier with about 1w tokens
## Pre-Training ways
   - using small tokenizer or large tokenizer
   - only mask edge 
   - only mask node
   - mask both node and edge


## Run

Please refer `utils/argument.py`

```shell
python main.py \
    --learning_rate 3e-5 \
    --do_train \
    --do_eval \
    --seed 42 \
    --epochs 5 \
    --batch_size 32 \
    --masked_edge \
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
    --tokenizer_dir 'tokenizer' \
    --token_vocab 'esperberto-vocab.json' \
    --token_merge 'esperberto-merges.txt' 2>&1 | tee  $saved_model/log.txt
```
Pretraining script: pretrain/jobs_masked_node_edge/run_bert_dim_384_6w_masked_node_edge.sh
test script:  downstream_tasks/none-phishing-scam-classfication/run_bert_dim_128_384_masked_node_masked_edge.sh
## Downstream tasks
We devided our data into three datasets, which are used to train, valid, test

n steps train, validation, -> save best model 

load best model - run test -> report
test


- downstream_tasks
    - none-phishing-scam-classfication
        - run.sh # run this script to start training, there should be different script for different models for each model has different settings. 
        - data # folder to store raw data 
        - processed # folder to store processed data
        - tokenizer # small tokenizer
        - tokenizer_larger # largee tokenizer
        - init.py
        - main.py
    - function_name_prediction
    - task3
