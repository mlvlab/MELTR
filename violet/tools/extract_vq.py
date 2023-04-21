
import argparse, base64, io, pickle
from glob import glob

from tqdm import tqdm

import numpy as np
import torch as T
import torchvision as TV
from dall_e import map_pixels, unmap_pixels, load_model

from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--frame', required=True, type=int)
    
    args = parser.parse_args()
    
    return args

def proc_buf(buf, _F):
    img = Image.open(io.BytesIO(base64.b64decode(buf)))
    w, h = img.size
    img = TV.transforms.Compose([TV.transforms.Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                                 TV.transforms.Resize([_F, _F]), 
                                 TV.transforms.ToTensor()])(img).unsqueeze(0)
    img = map_pixels(img)
    return img

if __name__=='__main__':
    args = get_args()
    
    dalle_enc = load_model('encoder.pkl', T.device('cpu')).cuda() # https://cdn.openai.com/dall-e/encoder.pkl
    # dalle_dec = load_model('decoder.pkl', T.device('cpu')).cuda() # https://cdn.openai.com/dall-e/decoder.pkl
    
    
    lst = glob(f'{args.path}/pickles/*.pkl')
    pickle_list = []
    for file in tqdm(lst):
        pickle_list.append(pickle.load(open(f'{file}', 'rb'))) 

    for pkl in tqdm(pickle_list):
        vq = {}
        for vid in pkl:
            imgs = [proc_buf(b, int(args.frame//32*8)) for b in pkl[vid]]
            imgs = T.cat(imgs, dim=0)
            
            z = dalle_enc(imgs.cuda())
            z = T.argmax(z, dim=1)
            vq[vid] = z.data.cpu().numpy().astype(np.int16)
            
            '''o = T.nn.functional.one_hot(z, num_classes=dalle_enc.vocab_size).permute(0, 3, 1, 2).float()
            o = dalle_dec(o).float()
            rec = unmap_pixels(T.sigmoid(o[:, :3]))
            rec = [TV.transforms.ToPILImage(mode='RGB')(r) for r in rec]'''
        pickle.dump(vq, open(f'{args.path}/vq/{vid}_vq.pkl', 'wb'))
    