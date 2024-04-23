
# All-in-one + MELTR


## Preparation
### Requirements

Our code is implemented under [All-in-one](https://github.com/showlab/all-in-one) environment with PyTorch 1.10+.

### Datasets

We use MSRVTT for text-to-video retrieval and All-in-one also provides downstream datasets [here](https://github.com/showlab/all-in-one/blob/main/DATA.md).

Annotation files of MSRVTT can be found [here](https://drive.google.com/drive/folders/1nXWGRKjm6fwYly4YCgdKu7XtV2IXGUix).

### Pretrained checkpoint

You can download the pretrained checkpoint of All-in-one [here](https://drive.google.com/file/d/1Yd2lKppaduqG_RO1gCA6OpAfB0_IXDoX/view?usp=sharing).

Then, place the files as follows:

```
data
 |─ msrvtt
 │   └─ videos
 |   |   │─ video0.mp4
 |   |   :
 │   │─ train_list_9k.txt
 |   │─ train_list_7k.txt
 |   │─ val_list_jsfusion.txt
 |   │─ MSR_VTT.json
 |   │─ jsfusion_val_caption_idx.pkl

checkpoint
 |─ all-in-one-plus-224.ckpt
```



## Training & Evaluation

```
python run.py with \
data_root=./data/msrvtt num_gpus=8 num_nodes=1 \
per_gpu_batchsize=16 msrvtt_retrieval_MELTR \
num_frames=3 \
load_path="./checkpoint/all-in-one-plus-224.ckpt"
```




## Acknowledgement
This repo is built upon [All-in-one](https://github.com/showlab/all-in-one).
