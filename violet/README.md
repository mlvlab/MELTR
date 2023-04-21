# Violet + MELTR


## Preparation

### Requirements

Our code is implemented under [Violet](https://github.com/tsujuifu/pytorch_violet) environment with PyTorch 1.10+.

### Datasets

We use three datasets (MSRVTT, TGIF, and MSVD). Violet also provides downstream datasets and annotation files [here](https://drive.google.com/drive/u/2/folders/1BisJHVUOLeHWmnAeMrCHvy1BP9XBXNkQ). 

Annotation files of msrvtt can be found [here](https://drive.google.com/drive/u/0/folders/11IHET_-sI7w5gq1f62EOEDWymv5RRhSi).

Download them and run the below command to extract VQ tokens for MVM.

```
cd tools
wget https://cdn.openai.com/dall-e/encoder.pkl # download trained dall-e encoder
python extract_vq.py --path=msrvtt --frame=224 # output: msrvtt_vq.pkl
```

### Pretrained checkpoint

You can download pretrained checkpoint of Violet [here](https://drive.google.com/file/d/1RLbthdRIflxCFjRTcVV5jQJGP30_lNfg/view).

Then, place the files as follows:

```
data
 |─ msrvtt
 │   |─ img_msrvtt.pkl
 │   │─ msrvtt_vq.pkl
 |   │─ train_9k.json
 |   │─ train_7k.json
 |   │─ test.json
 |
 |─ tgif
 |   │─ img_tgif.pkl
 |   │─ tgif_vq.pkl
 |   |─ txt_tgif-action.json
 |   |─ txt_tgif-transition.json
 |   |─ txt_tgif-frame.json
 |   
 |─ msvd
 |   │─ img_msvd.pkl
 |   │─ msvd_vq.pkl
 |   │─ txt_msvd-qa.json

checkpoint
 |─ ckpt_violet_pretrain.pt
```



## Training & Evaluation

+ Multiple-Choice Question Answering
```
python main_qamc.py ./args/args_tgif-action.json
python main_qamc.py ./args/args_tgif-transition.json
```
+ Open-Ended Question Answering
```
python main_qaoe.py ./args/args_msvd-qaoe.json
python main_qaoe.py ./args/args_tgif-frame.json
```
+ Text-to-Video Retrieval
```
python main_retrieval.py ./args/args_msrvtt-retrieval_7k.json
python main_retrieval.py ./args/args_msrvtt-retrieval_9k.json
python eval_retrieval.py ./args/args_msrvtt-retrieval_eval.json
```
You may modifty 'path_ckpt' of './args/args_msrvtt-retrieval_eval.json' for evaluation.



## Acknowledgement

This repo is built upon [Violet](https://github.com/tsujuifu/pytorch_violet).
