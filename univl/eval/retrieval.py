import torch
from tqdm import tqdm
import numpy as np
from utils.metrics import compute_metrics
from utils.utils import parallel_apply
global logger

def _run_on_single_gpu(net, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(tqdm(batch_list_t)):
        input_ids, input_mask, segment_ids, _, _, _, _, _, _ = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            _, _, _, video, video_mask, _, _, _, _ = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits = net.get_similarity_logits_align(sequence_output, visual_output, input_mask, video_mask)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_retrieval_epoch(model, test_dataloader, device, n_gpu, logger):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()
    with torch.no_grad():
        batch_list = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        for bid, batch in enumerate(tqdm(test_dataloader)):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, video, video_mask, _, _, _, _ = batch


            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

            batch_sequence_output_list.append(sequence_output)
            batch_visual_output_list.append(visual_output)
            batch_list.append(batch)

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list[s_:e_])
                    batch_list_v_splits.append(batch_list)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)
            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

        else:
            sim_matrix = _run_on_single_gpu(model, batch_list, batch_list, batch_sequence_output_list, batch_visual_output_list)
            sim_matrix = np.concatenate(sim_matrix, axis=0)

    metrics = compute_metrics(sim_matrix) # 53 * (64, 3369)
    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
    logger.info('\t>>>  R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.
                format(metrics['R1'], metrics['R5'], metrics['R10'], metrics['MR']))

    R1, R5, R10, MR = metrics['R1'], metrics['R5'], metrics['R10'], metrics['MR']

    return R1, R5, R10, MR
