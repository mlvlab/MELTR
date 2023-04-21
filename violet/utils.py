import torch 

class AverageMeter:
    ''' Computes and stores the average and current value. '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        if type(val) == torch.Tensor:
            val = float(val.detach().cpu().data)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def sample(self):
        return "\
        end = time.time() \n\
        batch_time = AverageMeter() \n\
        batch_time.update(time.time() - end) \n\
        end = time.time() \n\
        avg_score = AverageMeter()\n\
        accuracy = 0.1\n\
        avg_score.update(accuracy)\n\
        losses = AverageMeter()\n\
        loss = 0\n\
        batch_size = 128\n\
        losses.update(loss.data.item(), batch_size)\n\
        print(f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\n\
              f'loss {losses.val:.4f} ({losses.avg:.4f})\t' \n\
              f'acc {avg_score.val:.4f} ({avg_score.avg:.4f})')"