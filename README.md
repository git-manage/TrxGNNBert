## TrxGNNbert
This is the code and dataset for the paper "TRXGNNBERT: Transaction BERT with Graph Encoder"

##  Getting started
### 1.download dataset and pretrained model from 

- https://pan.baidu.com/s/17GIVex5uMnV8fNUwiRJDZg?pwd=1111&_at_=1729610977990#list/path=%2F
- You can just go to step 3 to train with the pretrained model we provide, or if you want to pretrain your own model with different setting, please refer to step 2. 

### 2. Pretrain TrxGNNBert Model
- Download the pretrain dataset folder, choose one of dataset in the folder.
- Extract the choosen dataset and rename the folder as 'data'
- Place the dataset under the pretrain profile
#### parameter options

##### Data_folder
-raw_data_folder : folder path to the *.pl files, `raw_data` for all all, `raw_data_6w` for 6w trx.

-pickle_path : save preprocessed graph daa
##### Tokenizer
- `tokenizer` is a small tokenizer with about 5,000 tokens
- `tokenizer_larger` is a large tokenzier with more than 1w tokens
##### Pre-Training parameter
   - mask edge # only mask edge features during pretraining 
   - mask node # only mask node features during pretraining
   - mask node and edge
##### Full parameter
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
#### RUN
bash  pretrain/jobs_masked_node_edge/jobs.sh 

### 3. Train TrxGNNBert Model

#### Dataset and downstream tasks

We provide three dataset in the link above, representing for three different downstream tasks.Among them, the 'EOA dataset' folder contains dataset for detection of suspicious Accounts, the 'EOA' folder contain dataset for Detection of Suspicious Accounts,the 'ternary classification dataset' and 'binary classification dataset' contains dataset for detection of suspicious transaction.

#### Training details
We divided each dataset into three part, train dataset, valid dataset, and test dataset.

After 1 epoch of training, we will save the best model in validation and begin to test. We train out model 10 epoch on every dataset. 
#### Folder structure of folder 'downstream_tasks'
- downstream_tasks
    - none-phishing-scam-classfication
        - run.sh #
        - data # folder to store dataset
        - processed # folder to store preprocessed dataset
        - tokenizer # small tokenizer
        - tokenizer_larger # larger tokenizer
        - init.py
        - main.py
    - function_name_prediction
    - task3
#### RUN
bash downstream_tasks/none-phishing-scam-classfication/run_bert_dim_128_384_masked_node_masked_edge.sh
