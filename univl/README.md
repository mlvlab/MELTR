# UniVL + MELTR

## Requirements

Our code is implemented under [UniVL](https://github.com/microsoft/UniVL) environment.

## Datasets

We use two datasets (msrvtt, youcook). UniVL also provides downstream datasets and annotation files [here](https://github.com/microsoft/UniVL/blob/main/dataloaders/README.md).
Annotation files of msrvtt that we used can be found [here](https://drive.google.com/drive/u/0/folders/1JmbjHymmALXzsowKjpNheRo3XApAe8Yh).

Note: As mentioned in UniVL, transcript is not publicly available due to legal issues.



### YoucookII

```
mkdir -p data
cd data
wget https://github.com/microsoft/UniVL/releases/download/v0/youcookii.zip
unzip youcookii.zip
cd ..
```

### MSRVTT

```
mkdir -p data
cd data
wget https://github.com/microsoft/UniVL/releases/download/v0/msrvtt.zip
unzip msrvtt.zip
cd ..
```


## Pretrained checkpoint

```
mkdir -p ./checkpoint
wget -P ./checkpoint https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
```

## Training & Evaluation

* Text-to-Video Retrieval on YouCook2

```
python main.py --expand_msrvtt_sentences --do_train --train_sim_after_cross \
--init_model ./checkpoint/univl.pretrained.bin --output_dir ckpts/RY \
--bert_model bert-base-uncased --do_lower_case --warmup warmup_linear_down \
--eval_task retrieval --datatype youcook --youcook_train_csv /path/to/data/youcookii_train.csv \
--youcook_val_csv /path/to/data/youcookii_val.csv \
--youcook_features_path /path/to/data/youcookii_videos_features.pickle \
--youcook_data_path /path/to/data/youcookii_data.transcript.pickle
```

* Text-to-Video Retrieval on MSRVTT-9K

```
python main.py --expand_msrvtt_sentences --do_train --train_sim_after_cross \
--init_model ./checkpoint/univl.pretrained.bin --output_dir ckpts/R9K  \
--bert_model bert-base-uncased --do_lower_case --warmup warmup_linear_down \
--eval_task retrieval --datatype msrvtt9K --VTT_train_csv /path/to/data/MSRVTT_train.9k.csv \
--VTT_val_csv /path/to/data/MSRVTT_JSFUSION_test.csv \
--VTT_data_path /path/to/data/MSRVTT_transcript_data_v2.json \
--VTT_features_path /path/to/data/msrvtt_videos_features.pickle
```

* Text-to-Video Retrieval on MSRVTT-7K

```
python main.py --expand_msrvtt_sentences --do_train --train_sim_after_cross \
--init_model ./checkpoint/univl.pretrained.bin --output_dir ckpts/R7K  \
--bert_model bert-base-uncased --do_lower_case --warmup warmup_linear_down \
--eval_task retrieval --datatype msrvtt7K --VTT_train_csv /path/to/data/MSRVTT_train.9k.csv \
--VTT_val_csv /path/to/data/MSRVTT_JSFUSION_test.csv \
--VTT_data_path /path/to/data/MSRVTT_transcript_data_v2.json \
--VTT_features_path /path/to/data/msrvtt_videos_features.pickle
```

* Captioning on YouCook2

```
python main.py --expand_msrvtt_sentences --do_train --train_sim_after_cross \
--init_model ./checkpoint/univl.pretrained.bin --output_dir ckpts/CY1 \
--bert_model bert-base-uncased --do_lower_case --warmup warmup_linear_down \
--eval_task caption --datatype youcook --youcook_train_csv /path/to/data/youcookii_train.csv \
--youcook_val_csv /path/to/data/youcookii_val.csv \
--youcook_features_path /path/to/data/youcookii_videos_features.pickle \
--youcook_data_path /path/to/data/youcookii_data.transcript.pickle
```

* Captioning on MSRVTT-Full

```
python main.py --expand_msrvtt_sentences --do_train --train_sim_after_cross \
--init_model ./checkpoint/univl.pretrained.bin --output_dir ckpts/CF \
--bert_model bert-base-uncased --do_lower_case --warmup warmup_linear_down \
--eval_task caption --datatype msrvttFull --VTT_train_csv /path/to/data/MSRVTT_train.9k.csv \
--VTT_val_csv /path/to/data/MSRVTT_JSFUSION_test.csv \
--VTT_data_path /path/to/data/MSRVTT_transcript_data_v2.json \
--VTT_features_path /path/to/data/msrvtt_videos_features.pickle
```



## Acknowledgement

This repo is built upon [UniVL](https://github.com/microsoft/UniVL).



