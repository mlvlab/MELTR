from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import (SequentialSampler)

from dataloaders.dataloader_msrvtt import MSRVTT_9K_Test, MSRVTT_FULL_Test, MSRVTT_7K_Train, MSRVTT_FULL_Train, MSRVTT_9K_Train
from dataloaders.dataloader_youcook import Youcook_Train, Youcook_Retrieval_Test, Youcook_Caption_Test

from utils.utils import str2list, AverageMeter, get_logger
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.tokenization import BertTokenizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from nlgeval import NLGEval
from tqdm import tqdm
import numpy as np
import argparse
import random
global logger
import socket
import torch
import os

from modules.meltr import MELTR, MELTROptimizer
from modules.optimization import BertAdam
from modules.modeling import UniVL

from eval.captioning import eval_caption_epoch
from eval.retrieval import eval_retrieval_epoch


def get_args(description='UniVL on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--youcook_train_csv',     type=str, default='/data/project/rw/youcookii_feature/youcookii_train.csv', help='')
    parser.add_argument('--youcook_val_csv',       type=str, default='/data/project/rw/youcookii_feature/youcookii_val.csv', help='')
    parser.add_argument('--youcook_features_path', type=str, default='/data/project/rw/youcookii_feature/youcookii_videos_features.pickle', help='feature path')
    parser.add_argument('--youcook_data_path',     type=str, default='/data/project/rw/youcookii_feature/youcookii_data.transcript_v4.pickle', help='data pickle file path')

    parser.add_argument('--VTT_train_csv',         type=str, default='/data/project/rw/msrvtt_feature/data/MSRVTT_train.9k.csv', help='')
    parser.add_argument('--VTT_val_csv',           type=str, default='/data/project/rw/msrvtt_feature/data/MSRVTT_JSFUSION_test.csv', help='')
    parser.add_argument('--VTT_data_path',         type=str, default='/data/project/rw/msrvtt_feature/data/MSRVTT_data.json', help='data pickle file path')
    parser.add_argument('--VTT_features_path',     type=str, default='/data/project/rw/msrvtt_feature/data/msrvtt_videos_features.pickle', help='feature path')

    parser.add_argument("--init_model", default="./univl.pretrained.bin", type=str, required=False, help="Initial model.")

    parser.add_argument('--num_thread_reader', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=64, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.01, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=10, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default="./ckpt/temp", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True,
                        help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--train_sim_after_cross', action='store_true', help="Test retrieval after cross encoder.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--is_distributed', default=1, type=int, help='')
    parser.add_argument('--tasks', default=[1, 1, 1, 1, 1, 1, 1, 1], type=str2list, help='task')
    parser.add_argument('--target_tasks', default=[0, 1, 0, 0, 0, 0, 0, 0], type=str2list, help='target task')
    parser.add_argument('--lr_vnet', default=0.001, type=float, help='Vnet Learning Rate')
    parser.add_argument('--decay_vnet', default=0.00003, type=float, help='Vnet weight decay')
    parser.add_argument('--ip_list', default=None, help="ipList", type=str2list)

    parser.add_argument('--warmup', default='warmup_linear', help="warmup_cosine, warmup_constant, warmup_linear, warmup_linear_down, none", type=str)
    parser.add_argument('--scheduler_epoch', default=4, help="", type=int)
    parser.add_argument('--milestones', default=None, help="", type=str2list)
    parser.add_argument('--milestones_proportion', default=0.1, help="", type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--transformer_dim', default=[512,128,256], type=str2list)

    parser.add_argument('--port', type=int, default=0, help='port')
    parser.add_argument('--auxgrad-every', type=int, default=3, help="number of opt. steps between aux params update")
    parser.add_argument('--hidden-dim', type=lambda s: [item.strip() for item in s.split(',')], default='10,10',
                        help="List of hidden dims for nonlinear")

    parser.add_argument('--eval_task',  default="retrieval", type=str, choices=["retrieval", "caption"], help="Evaluation Task")
    parser.add_argument('--eval_start', default=[100, 0], type=str2list, help='start evaluation')
    parser.add_argument('--eval_term',  default=[1,1], type=str2list, help='start evaluation')

    parser.add_argument("--datatype", default="msrvtt9K", choices=['youcook', 'msrvtt9K', 'msrvttFull', 'msrvtt7K'], type=str, help="Point the dataset `youcook` to finetune.")
    parser.add_argument('--mask_probability', default=0.15, type=float, help="")
    parser.add_argument('--vnet_max_grad', default=50, type=float, help="")
    parser.add_argument('--max_grad_norm', default=1., type=float, help="")
    parser.add_argument('--gamma', default=0.1, type=float, help="")
    parser.add_argument('--reg', default=0, type=int, help="")

    args = parser.parse_args() # auxgrad-every



    if args.port == 0:
        args.port = random.randint(0, 20000)
    if args.datatype == "youcook":
        args.n_display = 20
    else:
        args.n_display = 100

    if args.eval_term[0] == 0: args.eval_term[0] = 1
    if args.eval_term[1] == 0: args.eval_term[1] = 1
    args.target_tasks = [0, 1, 0, 0, 0, 0, 0, 0] if args.eval_task == "retrieval" else [0, 0, 0, 0, 0, 0, 1, 0]

    if args.mask_probability > 1:
        raise ValueError("Please Check mask_probability")

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger

    if args.seed:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    if args.is_distributed:
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)
        args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))


    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu



def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    if args.is_distributed:
        model.to(device)
    else:
        model.cuda()
    return model

def prep_optimizer(args, model, num_train_optimization_steps, num_train_warmup_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup, warmup_proportion=args.warmup_proportion,
                         milestones=args.milestones, milestones_proportion=args.milestones_proportion,
                         t_total=num_train_optimization_steps, t_warmup=num_train_warmup_steps, weight_decay=args.lr_decay,
                         max_grad_norm=args.max_grad_norm)

    if args.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def dataloader_youcook_train(args, tokenizer):
    youcook_dataset = Youcook_Train(
        csv=args.youcook_train_csv,
        data_path=args.youcook_data_path,
        features_path=args.youcook_features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        downstream=args.eval_task,
        p=args.mask_probability
    )

    if args.is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True
    )

    return dataloader, len(youcook_dataset), train_sampler

def dataloader_youcook_test(args, tokenizer):
    youcook_retrieval = Youcook_Retrieval_Test(
        csv=args.youcook_val_csv,
        data_path=args.youcook_data_path,
        features_path=args.youcook_features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )


    test_retrieval_sampler = SequentialSampler(youcook_retrieval)
    dataloader_youcook_retrieval = DataLoader(
        youcook_retrieval,
        sampler=test_retrieval_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    youcook_caption = Youcook_Caption_Test(
        csv=args.youcook_val_csv,
        data_path=args.youcook_data_path,
        features_path=args.youcook_features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    test_caption_sampler = SequentialSampler(youcook_caption)
    dataloader_youcook_caption = DataLoader(
        youcook_caption,
        sampler=test_caption_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII retrieval validation pairs: {}'.format(len(youcook_retrieval)))
        logger.info('YoucookII caption validation pairs: {}'.format(len(youcook_caption)))

    return dataloader_youcook_retrieval, len(youcook_retrieval), dataloader_youcook_caption, len(youcook_caption)

def dataloader_msrvtt_train(args, tokenizer):
    if args.datatype =="msrvtt9K":
        msrvtt_dataset = MSRVTT_9K_Train(
            csv_path=args.VTT_train_csv,
            json_path=args.VTT_data_path,
            features_path=args.VTT_features_path,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            unfold_sentences=args.expand_msrvtt_sentences,
            p = args.mask_probability
        )
    elif args.datatype == "msrvttFull":
        msrvtt_dataset = MSRVTT_FULL_Train(
            csv_path=args.VTT_train_csv,
            json_path=args.VTT_data_path,
            features_path=args.VTT_features_path,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            split_type="train",
            p=args.mask_probability,
            downstream = args.eval_task
        )
    elif args.datatype == "msrvtt7K":
        msrvtt_dataset = MSRVTT_7K_Train(
            csv_path=args.VTT_train_csv,
            json_path=args.VTT_data_path,
            features_path=args.VTT_features_path,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            split_type="train",
            p=args.mask_probability
        )
    else:
        raise argparse.ArgumentTypeError('Please Check Datatype')

    if args.is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler




def dataloader_msrvtt_test(args, tokenizer, split_type="test", ):
    if args.datatype =="msrvtt9K" or args.datatype == "msrvtt7K":
        msrvtt_testset = MSRVTT_9K_Test(
            csv_path=args.VTT_val_csv,
            features_path=args.VTT_features_path,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
        )
    elif args.datatype == "msrvttFull":
        msrvtt_testset = MSRVTT_FULL_Test(
            csv_path=args.VTT_val_csv,
            json_path=args.VTT_data_path,
            features_path=args.VTT_features_path,
            max_words=args.max_words,
            feature_framerate=args.feature_framerate,
            tokenizer=tokenizer,
            max_frames=args.max_frames,
            split_type=split_type,
        )
    else:
        raise argparse.ArgumentTypeError('Please Check Datatype')

    test_msrvtt_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt_test = DataLoader(
        msrvtt_testset,
        sampler=test_msrvtt_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )

    return dataloader_msrvtt_test, len(msrvtt_testset), dataloader_msrvtt_test, len(msrvtt_testset)




def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}".format(epoch if type_name=="" else type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model
def train_epoch(epoch, args, net, auxiliary_combine_net, train_dataloader, device, n_gpu, optimizer, meta_optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    net.train()

    log_step = args.n_display
    total_losses = AverageMeter()
    batchs, atten_maps, task_grads, task_losses = [], [], [], []

    for step, batch in enumerate(tqdm(train_dataloader)):
        if n_gpu == 1 and args.is_distributed:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        else:
            batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        losses = net(input_ids, segment_ids, input_mask, video, video_mask,
                     pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                     masked_video=masked_video, video_labels_index=video_labels_index,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                     output_caption_ids=pairs_output_caption_ids, tasks=args.tasks)

        loss = auxiliary_combine_net(torch.stack(losses).unsqueeze(1))

        loss.backward()
        total_losses.update(float(loss.detach().cpu()))
        if args.is_distributed:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        optimizer.step(epoch=epoch)
        optimizer.zero_grad()

        if (global_step) % args.auxgrad_every == 0:
            if len(batchs) > 0:
                input_ids, input_mask, segment_ids, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = random.choices(batchs)[0]
            meta_val_loss = 0.
            losses_pri = net(input_ids, segment_ids, input_mask, video, video_mask,
                                pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                                masked_video=masked_video, video_labels_index=video_labels_index,
                                input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                                output_caption_ids=pairs_output_caption_ids, tasks=args.target_tasks)
            meta_val_loss += sum(losses_pri)

            if len(batchs) > 0:
                input_ids, input_mask, segment_ids, video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = random.choices(batchs)[0]

            inner_loop_end_train_loss = 0.
            losses_train = net(input_ids, segment_ids, input_mask, video, video_mask,
                                pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                                masked_video=masked_video, video_labels_index=video_labels_index,
                                input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                                output_caption_ids=pairs_output_caption_ids, tasks=args.tasks)
            loss = auxiliary_combine_net(torch.stack(losses_train).unsqueeze(1))
            inner_loop_end_train_loss += loss


            phi = list(auxiliary_combine_net.parameters())
            W = [p for n, p in net.named_parameters()]

            if args.reg:
                meta_val_loss += args.gamma * torch.norm(sum(losses_train) - loss)

            meta_optimizer.step(
                val_loss=meta_val_loss,
                train_loss=inner_loop_end_train_loss,
                aux_params=phi,
                parameters=W,
            )

        batchs.append(batch)
        if len(batchs) > 10:
            batchs.pop(0)

        global_step += 1
        if global_step % log_step == 0 and local_rank == 0:
            logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f", epoch + 1,
                        args.epochs, step + 1,
                        len(train_dataloader), "-".join([str('%.8f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                        total_losses.avg)


    return total_losses, global_step



DATALOADER_DICT = {}
DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_train, "val":dataloader_youcook_test}
DATALOADER_DICT["msrvtt9K"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test}
DATALOADER_DICT["msrvttFull"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test}
DATALOADER_DICT["msrvtt7K"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test}

def main_worker(gpu, ngpus_per_node, args):
    global logger
    args.local_rank = gpu  # 0
    args.world_rank = args.local_rank + args.world_rank * ngpus_per_node
    if args.is_distributed:
        print("SETTING : ", args.ip, args.local_rank, args.world_rank, args.world_size)
        torch.distributed.init_process_group(backend="nccl", init_method='tcp://{}:{}'.format(args.ip, args.port), world_size=args.world_size, rank=args.world_rank)  # 8, 4
    else:
        print("Not Distributed")
    args = set_seed_logger(args)

    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = init_model(args, device, n_gpu, args.local_rank)

    if args.eval_task == "caption":
        nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)
    else:
        nlgEvalObj = None

    assert args.datatype in DATALOADER_DICT
    test_retrieval_dataloader, test_retrieval_length, test_caption_dataloader, test_caption_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)

    if args.local_rank == 0:
        logger.info("***** Running Retrieval test *****")
        logger.info("  Num examples = %d", test_retrieval_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_retrieval_dataloader))
        logger.info("***** Running Captioning test *****")
        logger.info("  Num examples = %d", test_caption_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_caption_dataloader))

    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs
    num_train_warmup_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                              / args.gradient_accumulation_steps) * args.scheduler_epoch

    coef_lr = args.coef_lr
    if args.init_model:
        coef_lr = 1.0
    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, num_train_warmup_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

    if args.do_eval:
        model = load_model(-1, args, n_gpu, device, model_file=args.init_model)
        if args.local_rank == 0:

            if args.eval_task == "retrieval":
                R1, R5, R10, MR = eval_retrieval_epoch(model, test_retrieval_dataloader, device, n_gpu, logger)
                print("Retrieval : R1 : {:0.4f}, R5 : {:0.4f}, R10 : {:0.4f}, MR : {:0.4f}".format(R1, R5, R10, MR))
            if args.eval_task == "caption":
                Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr = eval_caption_epoch(args, model, test_caption_dataloader, tokenizer, device, n_gpu, logger, nlgEvalObj=nlgEvalObj)
                print("Caption : Bleu_1 : {:0.4f}, Bleu_2 : {:0.4f}, Bleu_3 : {:0.4f}, Bleu_4 : {:0.4f}, METEOR : {:0.4f}, ROUGE_L : {:0.4f}, CIDEr : {:0.4f}"
                      .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr))
        return



    # ================
    # hyperparam model
    # ================
    auxiliary_combine_net = MELTR(t_dim=args.taskNum, f_dim = args.transformer_dim[0], i_dim = 1, h1_dim = args.transformer_dim[1], h2_dim = args.transformer_dim[2], o_dim=1).to(device)
    auxiliary_param = list(auxiliary_combine_net.parameters())

    meta_opt = torch.optim.Adam(auxiliary_param, lr=args.lr_vnet, weight_decay=args.decay_vnet)

    meta_optimizer = MELTROptimizer(
        meta_optimizer=meta_opt, max_grad_norm=args.vnet_max_grad
    )
    auxiliary_combine_net.eval()

    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

    global_step = 0
    best_score = [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    for epoch in range(args.epochs):
        if args.is_distributed:
            train_sampler.set_epoch(epoch)
        tr_loss, global_step = train_epoch(epoch, args, model, auxiliary_combine_net, train_dataloader, device, n_gpu, optimizer, meta_optimizer, scheduler, global_step, local_rank=args.local_rank)

        if args.world_rank == 0:
            print("Epoch {}/{} Finished, Train Loss: {:0.4f}".format(epoch + 1, args.epochs, tr_loss.avg))
            save_model(epoch, args, model)

            if epoch >= args.eval_start[0] and (epoch % args.eval_term[0] == 0 and args.eval_task == "retrieval"):
                R1, R5, R10, MR = eval_retrieval_epoch(model, test_retrieval_dataloader, device, n_gpu, logger)
                if best_score[0][0] <= R1:
                    save_model(epoch, args, model, type_name="best")
                    best_score[0][0] = R1
                    best_score[0][1] = R5
                    best_score[0][2] = R10
                    best_score[0][3] = MR

                print("Retrieval : Best R1 Accuracy : {:0.4f}, Current R1 Accuracy: {:0.4f}".format(best_score[0][0], R1))
            if epoch >= args.eval_start[1] and (epoch % args.eval_term[1] == 0 and args.eval_task == "caption"):
                Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr = eval_caption_epoch(args, model, test_caption_dataloader, tokenizer, device, n_gpu, logger, nlgEvalObj=nlgEvalObj)

                if best_score[1][4] <= Bleu_1:
                    save_model(epoch, args, model, type_name="best")
                    best_score[1][0] = Bleu_1
                    best_score[1][1] = Bleu_2
                    best_score[1][2] = Bleu_3
                    best_score[1][3] = Bleu_4
                    best_score[1][4] = METEOR
                    best_score[1][5] = ROUGE_L
                    best_score[1][6] = CIDEr

                print("Caption : Best BLEU_4 : {:0.4f}, Best METEOR : {:0.4f}, Current BLEU_4 Accuracy: {:0.4f}".format(best_score[1][3], best_score[1][4], Bleu_4))

def main():
    args = get_args()
    taskName = ["joint", "alignment", "mlm", "mfm", "m_joint", "m_align", "decoder", "m_decoder"]
    args.taskName = np.array(taskName)[np.array(args.tasks) == 1]
    args.taskNum = (np.array(args.tasks) == 1).sum()
    print("-"*10, args.taskName, "-"*10)


    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    this_ip = s.getsockname()[0]
    s.close()

    if args.ip_list: # multi-node
        ip_list = args.ip_list
        args.ip = ip_list[0]
    else: # single-node
        args.ip = this_ip
        ip_list = [args.ip]

    args.world_size = len(ip_list)  # 8
    for i, ip in enumerate(ip_list):
        if ip == this_ip:
            args.world_rank = i

    ngpus_per_node = torch.cuda.device_count()  # 8
    args.world_size = ngpus_per_node * args.world_size  # 8 * 8
    if args.is_distributed:
        print("WORLD SIZE : ", args.world_size)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) # args.world_size or ngpus_per_node
    else:
        main_worker(0, ngpus_per_node, args)

if __name__ == "__main__":
    main()