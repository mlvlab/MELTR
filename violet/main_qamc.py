
from lib import *
from dataset import Dataset_Base
from model import VIOLET_Base
from agent import Agent_Base, Agent_Base_MELTR
import pandas as pd
from utils import AverageMeter
from meltr import MELTR
import random

class Dataset_QAMC(Dataset_Base):
    def __init__(self, args, split):
        super().__init__(args)
        
        dataset = args['dataset']
        self.imgs = pickle.load(open(f'./_data/{dataset}/img_{dataset}.pkl', 'rb'))
        self.vq = pickle.load(open(f'./_data/{dataset}/{dataset}_vq.pkl', 'rb'))
        annotation = json.load(open(f"./_data/{dataset}/{args['annotation_file']}"))
        
        if split == 'train':
            self.data = annotation['train']
        else:
            self.data = annotation['test']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid = self.data[idx]['video']

        img = []
        for b in self.imgs[vid]:
            img.append(self.str2img(b).unsqueeze(0))
        img = T.cat(img, dim=0)
        
        txt, mask = [], []
        options = " ".join(self.data[idx][f"option_{i}"] for i in range(self.args['size_option']))
        for i in range(self.args['size_option']):
            t, m = self.str2txt(self.data[idx]['question'] + 'Options: ' + options + "Answer: " + self.data[idx][f'option_{i}'])
            txt.append(t), mask.append(m)
        txt, mask = np.array(txt, dtype=np.int64), np.array(mask, dtype=np.int64)

        vq = np.array(sum([[-1] + c.flatten().tolist() for c in self.vq[vid]], []), dtype=np.int64)
        
        return img, txt, mask, vq, self.data[idx]['answer']

class VIOLET_QAMC(VIOLET_Base):
    def __init__(self):
        super().__init__()
        
        self.fc_mc = T.nn.Sequential(*[T.nn.Dropout(0.1), 
                                    T.nn.Linear(768, 768*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(768*2, 1)])
        
        self.fc_vtm = T.nn.Sequential(*[T.nn.Dropout(0.1), 
                                    T.nn.Linear(768, 768*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(768*2, 1)])
        
        bert = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.fc_mtm = bert.cls
        self.fc_mvm = T.nn.Sequential(*[T.nn.Dropout(0.1),
                                        T.nn.Linear(768, 768 * 2), T.nn.ReLU(inplace=True),
                                        T.nn.Linear(768 * 2, 8192)])

    def go_feat(self, img, txt, mask, m_txt):
        feat_img, mask_img = self.enc_img(img)
        feat_txt, mask_txt = self.enc_txt(txt), mask
        feat_m_txt, mask_m_txt = self.enc_txt(m_txt), mask
        return feat_img, mask_img, feat_txt, mask_txt, feat_m_txt, mask_m_txt
    
    
    def forward(self, img, txt, mask, m_txt):
        (_B, _T, _, _H, _W), (_, _O, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32
        
        feat_img, mask_img, feat_txt, mask_txt, feat_m_txt, mask_m_txt = self.go_feat(img, txt.flatten(0, 1), mask.flatten(0, 1), m_txt.flatten(0, 1))
        feat_img, mask_img = [feat_img.unsqueeze(1).expand([-1, _O, -1, -1]).flatten(0, 1), 
                              mask_img.unsqueeze(1).expand([-1, _O, -1]).flatten(0, 1)]
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        out_mc = self.fc_mc(out[:, (1+_h*_w)*_T, :]).squeeze().view([_B, _O])
        
        out, _ = self.go_cross(feat_img, mask_img, feat_m_txt, mask_m_txt)
        out_mtm, out_mvm = self.fc_mtm(out[:, (1 + _h * _w) * _T:]), self.fc_mvm(out[:, :(1 + _h * _w) * _T])
        
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        for i in range(_B):
            for j in range(_B):
                pdt_feat_img.append(feat_img[i].unsqueeze(0)), pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(feat_txt[j].unsqueeze(0)), pdt_mask_txt.append(mask_txt[j].unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [T.cat(x, dim=0) for x in [pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt]]
        
        out, _ = self.go_cross(pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt)
        out_vtm = self.fc_vtm(out[:, (1+_h*_w)*_T, :]).squeeze().view([_B, _B]) / 0.05
        
        ans_vtm = T.tensor([i for i in range(_B)]).long().cuda()
        
        return out_mtm, out_mvm, out_vtm, out_mc, ans_vtm   
        
class Agent_QAMC(Agent_Base_MELTR):
    def __init__(self, args, model, aux_model):
        super().__init__(args, model, aux_model)
        self.gamma = args['gamma']
    
    def masking(self, img, txt, vq):
        (_B, _T, _, _H, _W), (_, _O, _X) = img.shape, txt.shape
        _h, _w = _H // 32, _W // 32
        spc_txt = T.logical_or(T.logical_or(txt == 101, txt == 102), txt == 0)

        ans_mtm, ans_mvm = T.ones(txt.shape).long() * -1, T.ones(vq.shape).long() * -1
        for i in range(_B):
            mask_mtm = T.where(T.logical_and(T.logical_not(spc_txt[i]), T.rand(_X) < 0.15))[0]
            while len(mask_mtm) == 0:
                mask_mtm = T.where(T.logical_and(T.logical_not(spc_txt[i]), T.rand(_X) < 0.15))[0]

            mask_mvm = set()
            for _ in range(_T):
                t, h, w = [np.random.randint(1, _T) if _T > 1 else 1,
                           np.random.randint(1, _h * 2 // 3), np.random.randint(1, _w * 2 // 3)]
                t1, h1, w1 = [np.random.randint(0, _T - t + 1),
                              np.random.randint(0, _h - h + 1), np.random.randint(0, _w - w + 1)]
                for i_t in range(t1, t1 + t):
                    for i_h in range(h1, h1 + h):
                        for i_w in range(w1, w1 + w):
                            mask_mvm.add((i_t, i_h, i_w))
            mask_mvm = list(mask_mvm)

            for p in mask_mtm:
                ans_mtm[i][p], txt[i][p] = txt[i][p], 103
            
            cov = T.zeros(_T, _h, _w)
            for i_t, i_h, i_w in mask_mvm:
                cov[i_t][i_h][i_w] = 1.0
                p = (1 + _h * _w) * i_t + 1 + i_h * _w + i_w
                ans_mvm[i][p] = vq[i][p]
            cov = cov.unsqueeze(1).unsqueeze(3).unsqueeze(5).expand([-1, 3, -1, 32, -1, 32])
            cov = cov.flatten(2, 3).flatten(3, 4)
            img[i] *= (1.0 - cov)

        return img, txt, ans_mtm, ans_mvm


    def step(self, img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt, pri=False):
        img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt = [x.cuda() for x in [img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt]]
        self.optzr.zero_grad()
        
        out_mtm, out_mvm, out_vtm, out_mc, ans_vtm = self.model(img, txt, mask, m_txt)

        out_mvm = out_mvm.reshape(ans_mc.shape[0], -1, out_mvm.shape[1], out_mvm.shape[2])
        index = ans_mc[:, None, None, None].repeat(1, 1, out_mvm.shape[2], out_mvm.shape[3])
        out_mvm = T.gather(out_mvm, 1, index).squeeze(1)

        ls_mc = self.loss_func(out_mc, ans_mc)
        
        ls_vtm, ls_mtm, ls_mvm = [self.loss_func(o.flatten(0, len(o.shape) - 2), a.flatten(0, len(a.shape) - 1)) 
                                  for o, a in zip([out_vtm, out_mtm, out_mvm], [ans_vtm, ans_mtm, ans_mvm])]

        if pri:
            return [ls_mc]
        else:
            return [ls_mc, ls_vtm, ls_mtm, ls_mvm]
    
    def test(self, dl):
        ret = []
        for img, txt, mask, vq, ans_mc in tqdm(dl, ascii=True):
            m_img, m_txt, ans_mtm, ans_mvm = self.masking(img.detach().clone(), txt.detach().clone(), vq)
            out_mtm, out_mvm, out_vtm, out_mc, ans_vtm = self.model(img, txt, mask, m_txt)
            out = T.argmax(out_mc, dim=1)
            ac = (out == ans_mc.cuda()).float().mean().item()
            ret.append(ac)
        ret = float(np.average(ret))
        
        return ret
    
    def train(self, dl, args, e, global_step):
        aux_loss_0, aux_loss_1, aux_loss_2, aux_loss_3, pri_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        batchs, m_ans_mtm = [], T.tensor([0])
        for i, (img, txt, mask, vq, ans_mc) in enumerate(tqdm(dl)):
    
            img, m_txt, ans_mtm, ans_mvm = self.masking(img, txt.detach().clone(), vq)
            
            batch = [img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt]

            losses = self.step(img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt)

            loss = self.aux_model(T.stack(losses).unsqueeze(1).unsqueeze(0))        
            
            aux_loss_0.update(losses[0].item())
            aux_loss_1.update(losses[1].item())
            aux_loss_2.update(losses[2].item())
            aux_loss_3.update(losses[3].item())
            pri_loss.update(loss.item())


            self.scaler.scale(loss).backward()
            self.scaler.step(self.optzr)
            self.scaler.update()

            if (global_step) % args['auxgrad_every'] == 0:

                if len(batchs) > 0:
                    img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt = random.choices(batchs)[0]

                losses_pri = self.step(img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt, pri=True)
                meta_val_loss = sum(losses_pri)

                if len(batchs) > 0:
                    img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt = random.choices(batchs)[0]

                losses_train = self.step(img, txt, mask, ans_mc, ans_mtm, ans_mvm, m_txt)
                loss = self.aux_model(T.stack(losses_train).unsqueeze(1).unsqueeze(0))
                inner_loop_end_train_loss = loss 
                
                meta_val_loss = meta_val_loss + self.gamma * T.abs(sum(losses_train) - loss)
                
                phi = list(self.aux_model.parameters())
                W = [p for n, p in self.model.named_parameters()]
                self.meta_optim.step(val_loss=meta_val_loss, train_loss=inner_loop_end_train_loss, aux_params=phi, parameters=W)
            batchs.append(batch)
            if len(batchs) > 10:
                batchs.pop(random.choices(range(10), weights=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0])

            global_step += 1


        return aux_loss_0, aux_loss_1, aux_loss_2, aux_loss_3, pri_loss
    
if __name__=='__main__':
    args = json.load(open(sys.argv[1], 'r'))
    args['size_batch'] = args['size_batch']*T.cuda.device_count()
    args['path_output'] = 'checkpoint/_%s_%s'%(args['task'], datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(args['path_output'], exist_ok=True)
    json.dump(args, open('%s/args.json'%(args['path_output']), 'w'), indent=2)
    print(args)
    
    dl_tr, dl_ts = [T.utils.data.DataLoader(Dataset_QAMC(args, split), 
                                            batch_size=args['size_batch'], shuffle=(split=='train'), 
                                            num_workers=32, pin_memory=True, drop_last=True) for split in ['train', 'test']]
    
    log = {'ls_tr': [], 'ac_ts': []}
    json.dump(log, open('%s/log.json'%(args['path_output']), 'w'), indent=2)
    
    model = T.nn.DataParallel(VIOLET_QAMC().cuda())
    model.module.load_ckpt(args['path_ckpt'])
    T.save(model.module.state_dict(), '%s/ckpt_violet_%s_0.pt'%(args['path_output'], args['task']))
    
    auxnet_config = dict(t_dim=4, f_dim=256, i_dim=1, h1_dim=128, h2_dim=256, o_dim=1)
    aux_model = T.nn.DataParallel(MELTR(**auxnet_config).cuda())
    
    agent = Agent_QAMC(args, model, aux_model)
    global_step = 1
    for e in tqdm(range(args['size_epoch']), ascii=True):
        model.train()
        aux_loss_0, aux_loss_1, aux_loss_2, aux_loss_3, pri_loss = agent.train(dl_tr, args, e, global_step)
        
        model.eval()
        ac_ts = agent.test(dl_ts)
        
        T.save(model.module.state_dict(), '%s/ckpt_violet_%s_%d.pt'%(args['path_output'], args['task'], e+1))
        print('Ep %d: %.6f %.6f'%(e+1, pri_loss.avg, ac_ts))
        