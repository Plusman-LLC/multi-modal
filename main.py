import argparse
import copy
import csv
import os, glob, sys
import random
import torch
import tqdm
import cv2
import numpy as np
import pandas as pd
import pickle
import torch.nn.functional as F
import pandas as pd
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from nets import nn
from nets import vit
from nets import HALF_UDEN
from utils import util
from torch.distributions.beta import Beta
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

import monai
from monai.transforms import apply_transform,Compose, AddChanneld, ScaleIntensityRanged, ScaleIntensityd,CropForegroundd,     RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, ToTensord
from monai.transforms import RandShiftIntensityd,RandRotated, RandFlipd, RandZoomd, Rand3DElasticd
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.utils import set_determinism
from datetime import datetime
# from adabelief_pytorch import AdaBelief

# np.random.seed(seed=9999)
monai.config.print_config()
data_dir = os.path.join('.', 'train_data')
batch_size = 8#256

label_names = ['negative','positive','-1']

pos_list = [[_,1] for _ in glob.glob(data_dir+'/1/*')]
neg_list = [[_,0] for _ in glob.glob(data_dir+'/0/*')]

perm_pos = np.random.permutation(len(pos_list))
perm_neg = np.random.permutation(len(neg_list))
num_train_pos = int(len(pos_list)*0.5)
num_train_neg = int(len(neg_list)*0.5)
num_val_pos = int(len(pos_list)*0.3)
num_val_neg = int(len(neg_list)*0.3)
num_test_pos = len(pos_list) - num_train_pos - num_val_pos
num_test_neg = len(neg_list) - num_train_neg - num_val_neg

def calc_cmx(ys, ts):
    n_seg = 2
    ind = np.where(ts>-1)
    a = n_seg * ts[ind].reshape(-1).astype("i") + ys[ind].reshape(-1).astype("i")
#    a = a[ts.reshape(-1)>-1]
    a = np.concatenate([a, np.arange(0,n_seg**2)]) # to fill all combination
    a = np.bincount(a) - np.ones(n_seg**2) # subtract adjustment above
    cmx = a.reshape((n_seg, n_seg)).astype("i")
    return cmx

class dataset_lymph(torch.utils.data.Dataset):
    def __init__(self,is_train=True,transform=None,data_list=None,woagesexsmoking=False):
        self.is_train = is_train
        self.images, self.masks, self.labels0, self.tokens = [], [], [], []
        self.agesexsmoking = []
        self.woagesexsmoking = woagesexsmoking
        def read_input(path2,label0):
            nodule_crops = []
            mask_crops = []
            tokens = []
            try:
                with open(path2+'/nodule_crop.pkl', 'rb') as f:
                    nodule_crops += [pickle.load(f)]
                with open(path2+'/mask_crop.pkl', 'rb') as f:
                    mask_crops += [pickle.load(f)]
                with open(path2+'/token.pkl', 'rb') as f:
                    token = [pickle.load(f)]
                    token2 = [t.astype('f')*0 for t in token]
                    tokens += token2
                agesexsmoking = np.array([token[0][0]/100,token[0][1]],dtype=np.float32)
                if self.woagesexsmoking:
                    agesexsmoking *= 0
            except:
                print('error in ')
                return
            
            self.images.append(nodule_crops)
            self.masks.append(mask_crops)# .append(msk[None].astype(np.float32))
            self.tokens.append(tokens)
            self.labels0.append(np.array(label0,dtype=np.int32))
            self.agesexsmoking.append(agesexsmoking)
            
        for d in data_list:
            read_input(d[0],d[1])

        print()
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        _images = self.images[index].copy()
        _masks = self.masks[index].copy()
        label0 = self.labels0[index].copy()
        _tokens = self.tokens[index].copy()
        agesexsmoking = self.agesexsmoking[index].copy()
        
        if self.transform is not None:
            images, masks, tokens = [],[],[]
            _perm = np.random.permutation(range(len(_images)))
            for p in _perm:
                image, mask, token = _images[p], _masks[p], _tokens[p]
                d = {'image':image[None].astype('f')+1024, 'mask':mask[None].astype('f')}
                d = apply_transform(self.transform, d)
                images.append(d['image'][0]-1024)
                masks.append(d['mask'][0])
                tokens.append(token+np.random.randn(2)*0.05)
        else:
            images = _images
            masks = _masks
            tokens = _tokens
            
        _data = {'images':images, 'masks':masks, 'label0':label0, 'tokens':tokens, 'agesexsmoking':agesexsmoking}
        return _data

def count_pos_neg(t):
    num_pos = torch.maximum(torch.IntTensor([1]).cuda(),torch.sum(t>0))
    num_neg = torch.maximum(torch.IntTensor([1]).cuda(),torch.sum(t==0))
    return num_pos[0], num_neg[0]

def count_label(t,num_label):
    temp_list = [torch.maximum(torch.IntTensor([1]).cuda(),torch.sum(t==i)) for i in range(num_label)]
    return torch.cat(temp_list,dim=-1)
    
def contrast(CT,low,high):
    return ((torch.minimum(torch.FloatTensor([high]).cuda(),torch.maximum(CT,torch.FloatTensor([low]).cuda()))-low)/(high-low))
    
def batch(images, tokens, masks, label0, agesexsmoking, model_seg, model_vit, criterion_mask=None, criterion_label0=None):
    loss = 0.0
    label0 = label0.cuda()
    xs, dices = [], []
    for image,token,mask in zip(images,tokens,masks):
        image = contrast(image[None].cuda(),-57,164).float()
        token = token.float().cuda()
        mask = mask.cuda()
        mask = torch.maximum(torch.IntTensor([0]).cuda(),torch.minimum(torch.IntTensor([1]).cuda(),(mask+0.5).long()))
        shape = image.shape
        if criterion_mask:
            with torch.cuda.amp.autocast():
                y,h=model_seg(image)
                loss += criterion_mask(y, mask)
                x0=F.avg_pool3d(h,h.shape[2:])[:,:,0,0,0]#.squeeze()
                x1=F.max_pool3d(h,h.shape[2:])[:,:,0,0,0]#.squeeze()
                x = torch.cat([x0,x1],dim=1)
                xs.append(x)
        else:
            y,h=model_seg(image)
            x0=F.avg_pool3d(h,h.shape[2:])[:,:,0,0,0]#.squeeze()
            x1=F.max_pool3d(h,h.shape[2:])[:,:,0,0,0]#.squeeze()
            x = torch.cat([x0,x1],dim=1)
            xs.append(x)
            y,mask = F.softmax(y,1)[:,1].detach().cpu().numpy()>0.5, mask.cpu().numpy()
            cmx = calc_cmx(y,mask)
            dices.append(2*cmx[1,1]/(2*cmx[1,1]+cmx[0,1]+cmx[1,0]+1))
            
    xs = torch.cat(xs,dim=0)
    if criterion_mask:
        with torch.cuda.amp.autocast():
            out0, est0, cert0 = model_vit(xs,agesexsmoking)
            if label0.cpu().numpy()[0] != -1:
                loss += criterion_label0(out0, label0)#/batch_size
            return loss
    else:
        output0 = model_vit(xs,agesexsmoking)
        acc0 = util.accuracy(output0, label0, top_k=(1, 2))
        return np.mean(dices), acc0[0], output0

def train(args, perm_pos, perm_neg):
    epochs = 500
    cycle = 100
    util.set_seeds(args.rank)
    model_seg = HALF_UDEN.HalfUNET(1,2).cuda()
    model_vit = vit.ViT(num_classes0=2, dim=128, depth=6, heads=8, mlp_dim=128, pool = 'cls', channels = 64*2+2, dim_head = 64, dropout = 0., emb_dropout = 0.).cuda()
    if args.adam:
        lr = 0.0001#batch_size * torch.cuda.device_count() * 0.256 / 4096
        optimizer0 = torch.optim.AdamW(model_seg.parameters(), lr)# / 16)
        optimizer1 = torch.optim.AdamW(model_vit.parameters(), lr)# / 16*16)
    elif args.sgd: 
        lr = 0.001#0.1
        momentum = 0.9
        optimizer0 = torch.optim.SGD(model_seg.parameters(), lr=lr, momentum=momentum)
        optimizer1 = torch.optim.SGD(model_vit.parameters(), lr=lr, momentum=momentum)
    # elif args.adabelief: 
    #     lr = 1e-3
    #     optimizer0 = AdaBelief(model_seg.parameters(), lr=lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
    #     optimizer1 = AdaBelief(model_vit.parameters(), lr=lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
    else:
        lr = batch_size * torch.cuda.device_count() * 0.256 / 4096
        optimizer0 = nn.RMSprop(util.add_weight_decay(model_seg), lr, 0.9, 1e-3, momentum=0.9)
        optimizer1 = nn.RMSprop(util.add_weight_decay(model_vit), lr, 0.9, 1e-3, momentum=0.9)

    weights_path = 'weights'+('/adam' if args.adam else '/RMSprop')+('/woagesexsmoking' if args.woagesexsmoking else '')
    os.makedirs(weights_path,exist_ok=True)

    if args.distributed:
        model_seg = torch.nn.parallel.DistributedDataParallel(model_seg, device_ids=[args.local_rank])
        model_vit = torch.nn.parallel.DistributedDataParallel(model_vit, device_ids=[args.local_rank])
    else:
        model_seg = torch.nn.DataParallel(model_seg)
        model_vit = torch.nn.DataParallel(model_vit)
    criterion_mask = nn.CrossEntropyLoss2(weight=torch.FloatTensor([1,1]),epsilon=0.01).cuda()
    criterion_label0 = nn.CrossEntropyLoss2(weight=torch.FloatTensor([1,1]),epsilon=0.01).cuda()

    scheduler0 = torch.optim.lr_scheduler.OneCycleLR(optimizer0, max_lr=lr, total_steps=cycle)
    scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer1, max_lr=lr, total_steps=cycle)

    last_name = 'last_pt_1dep'
    best_name = 'best_pt_1dep'
    step_name = 'step_pt_1dep'
    
    train_transforms = Compose([
        RandShiftIntensityd(keys=['image'], prob=0.5, offsets=2),
        RandRotated(keys=['image', 'mask'], 
                    range_x=np.pi/2, 
                    range_y=np.pi/2, 
                    range_z=np.pi/2, 
                    padding_mode = ("zeros","zeros"),
                    prob=0.5, 
                    keep_size=False, 
                    mode=('bilinear', 'nearest')),
        RandFlipd(keys=['image', 'mask'],prob=0.5),
        RandZoomd(keys=['image', 'mask'],
                  prob=0.5, min_zoom=0.95, 
                  max_zoom=1.05,
                  padding_mode = ("zeros","zeros"),
                  keep_size = False, 
                  mode=('trilinear', 'nearest')),
        RandAffined(keys=['image', 'mask'], mode=('bilinear', 'nearest'), prob=0.5, 
                    rotate_range=(np.pi/12, np.pi/12, np.pi/12), 
                    padding_mode = ("zeros","zeros"),
                    scale_range=(0.05, 0.05, 0.05)),
    ])
    
    data_list = np.array(pos_list)[perm_pos[:num_train_pos]].tolist()
    data_list += np.array(neg_list)[perm_neg[:num_train_neg]].tolist()
    train_ds = dataset_lymph(is_train=True,transform=train_transforms,data_list=data_list,woagesexsmoking=args.woagesexsmoking)
    train_ld = monai.data.DataLoader(train_ds, batch_size=1, shuffle=True)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        sampler = None

    with open(weights_path+f'/{step_name}.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@0', 'acc@1','dice'])
            writer.writeheader()
        best_auc0, best_auc1, best_auc2, best_auc3, best_dice, epoch_at_best, train_auc0_atbest = 0, 0, 0, 0, 0, 0, 0
        test_auc0, test_auc1, test_auc2, test_auc3 = 0, 0, 0, 0
        for epoch in range(0, epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                bar = tqdm.tqdm(train_ld, total=len(train_ld))
            else:
                bar = train_ld
            model_seg.train()
            model_vit.train()
            iterator = train_ld.__iter__()
            step=0
            for itr in range(0,len(train_ds),batch_size):
                itr_to = min(len(train_ds),itr+batch_size)
                step += 1
                images, masks,tokens = [],[],[]
                loss = 0.0
                for b in range(itr,itr_to):
                    d = iterator.__next__()
                    images,masks,label0,tokens,agesexsmoking = d['images'], d['masks'], d['label0'].long(), d['tokens'], d['agesexsmoking']
                    loss += batch(images, tokens, masks, label0, agesexsmoking, model_seg, model_vit, criterion_mask, criterion_label0)/(itr_to-itr)

                scheduler0.optimizer.zero_grad()
                scheduler1.optimizer.zero_grad()
                loss.backward()
                scheduler0.optimizer.step()
                scheduler1.optimizer.step()
    
                torch.cuda.synchronize()
                if args.local_rank == 0:
                    bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))

            scheduler0.step()
            scheduler1.step()
            if (epoch+1)%cycle == 0:
                optimizer0 = torch.optim.AdamW(model_seg.parameters(), lr)# / 16)
                optimizer1 = torch.optim.AdamW(model_vit.parameters(), lr)# / 16*16)
                if args.sgd: 
                    optimizer0 = torch.optim.SGD(model_seg.parameters(), lr=lr, momentum=momentum)
                    optimizer1 = torch.optim.SGD(model_vit.parameters(), lr=lr, momentum=momentum)
                scheduler0 = torch.optim.lr_scheduler.OneCycleLR(optimizer0, max_lr=lr, total_steps=cycle)
                scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer1, max_lr=lr, total_steps=cycle)

            if args.local_rank == 0:
                acc0, dice, auc0, num0,_,_ = test(args, perm_pos[num_train_pos:num_train_pos+num_val_pos], perm_neg[num_train_neg:num_train_neg+num_val_neg], model_seg.eval(),model_vit.eval())
                writer.writerow({'acc@0': str(f'{acc0:.3f}'),
                                 'dice': str(f'{dice:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                state_s = {'model_seg': copy.deepcopy(model_seg).half()}
                torch.save(state_s, weights_path+f'/{last_name}_seg.pt')
                state = {'model_vit': copy.deepcopy(model_vit).half()}
                torch.save(state, weights_path+f'/{last_name}_vit.pt')

                _, _, curr_auc0, num0, _, _ = test(args, perm_pos[num_train_pos+num_val_pos:], perm_neg[num_train_neg+num_val_neg:], model_seg.eval(),model_vit.eval())
                _, _, train_auc0, _, _, _ = test(args, perm_pos[:num_train_pos], perm_neg[:num_train_neg], model_seg.eval(),model_vit.eval())

                save_ = False
                if auc0 > best_auc0 and (epoch>20) and (train_auc0>0.75):
                    save_ = True
                elif auc0 == best_auc0:
                    # if dice > best_dice:
                    if train_auc0 > train_auc0_atbest:
                        save_ = True

                if save_:
                    epoch_at_best = epoch+1
                    print('save best state')
                    torch.save(state, weights_path+f'/{best_name}_vit.pt')
                    torch.save(state_s, weights_path+f'/{best_name}_seg.pt')
                    _, _, test_auc0, _, lab1, pred1 = test(args, perm_pos[num_train_pos+num_val_pos:], perm_neg[num_train_neg+num_val_neg:], model_seg.eval(),model_vit.eval(),num_trail=5)
                    best_auc0 = auc0 #max(auc0, best_auc0)
                    best_dice = dice #max(dice, best_dice)
                    train_auc0_atbest = train_auc0
                    
                del state_s
                del state
                
                print(' val auc@best: %10.3g at %10.3g' % (best_auc0, epoch_at_best))
                print('test auc@best: %10.3g at %10.3g' % (test_auc0, epoch_at_best))
                print('    train auc: %10.3g at %10.3g' % (train_auc0, epoch+1))
                print('      val auc: %10.3g at %10.3g' % (auc0, epoch+1))
                print('     test auc: %10.3g at %10.3g' % (curr_auc0, epoch+1))
                print('     num test: %10.3g' % (num0))
                
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    return best_dice, best_auc0, epoch_at_best, train_auc0_atbest, test_auc0, num0, lab1, pred1

def test(args, idxs_pos, idxs_neg, model_seg=None, model_vit=None, num_trail=5):
    weights_path = 'weights'+('/adam' if args.adam else '/RMSprop')+('/woagesexsmoking' if args.woagesexsmoking else '')
    if model_seg is None:
        model_seg = torch.load(weights_path+'/best_pt_seg.pt', map_location='cuda')['model_seg'].float().eval()
        model_vit = torch.load(weights_path+'/best_pt_vit.pt', map_location='cuda')['model_vit'].float().eval()
    
    data_list = np.array(pos_list)[perm_pos[idxs_pos]].tolist()
    data_list += np.array(neg_list)[perm_neg[idxs_neg]].tolist()
    test_ds = dataset_lymph(is_train=False,transform=None,data_list=data_list)
    test_ld = monai.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    dices = util.AverageMeter()
    result,lab0,pred0,lab1,pred1,lab2,pred2,lab3,pred3 = [],[],[],[],[],[],[],[],[]
    with torch.no_grad():
        step=0
        iterator = test_ld.__iter__()
        for itr in range(0,len(test_ds),batch_size):
            itr_to = min(len(test_ds),itr+batch_size)
            step += 1
            # images, masks, tokens = [],[],[]
            for b in range(itr,itr_to):
                d = iterator.__next__()
                # print(d['label0'])
                _images,_masks,label0,_tokens,agesexsmoking = d['images'], d['masks'], d['label0'].long(),d['tokens'],d['agesexsmoking']
                if False:#token_shuffle:
                    prob0,prob1,prob2,prob3=0.0,0.0,0.0,0.0
                    for l in range(num_trail):
                        _perm = np.random.permutation(range(len(_images)))
                        images, masks, tokens = [],[],[]
                        for p in _perm:
                            image, mask, token = _images[p], _masks[p], _tokens[p]
                            images.append(image)
                            masks.append(mask)
                            tokens.append(token)
    
                        dice, acc0, output0 = batch(images, tokens, masks, label0, agesexsmoking, model_seg, model_vit)
                        prob0 += F.softmax(output0,dim=1).cpu().numpy()[0,1]
                        torch.cuda.synchronize()
                    prob0,prob1,prob2,prob3=prob0/num_trail,prob1/num_trail,prob2/num_trail,prob3/num_trail
                else:
                    images = _images
                    masks = _masks
                    tokens = _tokens
                    
                    dice, acc0, output0 = batch(images, tokens, masks, label0, agesexsmoking, model_seg, model_vit)
                    prob0 = F.softmax(output0,dim=1).cpu().numpy()[0,1]
                    torch.cuda.synchronize()

                if label0.cpu().numpy()[0]!=-1:
                    top1.update(acc0.item(), len(images))#.size(0))
                    lab0.append(label0.cpu().numpy()[0])
                    pred0.append(prob0)
                dices.update(dice, len(images))#.size(0))
                result.append([step,label0.item(),acc0.item(),dice,prob0])
        acc0, acc1, dice = top1.avg, top5.avg, dices.avg
        fpr, tpr, thresholds = roc_curve(np.array(lab0),np.array(pred0))
        auc0 = auc(fpr, tpr)
        df = pd.DataFrame(result)#,columns=['id'])
        df.to_csv('test_result_{}.csv'.format('adam' if args.adam else 'sgd'),index=False)

    if model_seg is None:
        torch.cuda.empty_cache()
    else:
        return acc0, dice, auc0, len(lab0), lab0, pred0


def print_parameters(args):
    model = nn.EfficientNet(args,num_class=4).eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {int(params)}')

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--lr', action='store_true') 
    parser.add_argument('--adam', action='store_true') 
    parser.add_argument('--sgd', action='store_true') 
    # parser.add_argument('--adabelief', action='store_true') 
    parser.add_argument('--tophat', action='store_true') 
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--woagesexsmoking', action='store_true')

    args = parser.parse_args()
    args.distributed = False
    args.rank = 0
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    if args.local_rank == 0:
        print_parameters(args)
    if args.test:
        test(args)
    if args.cv:
        if args.score:
            f0 = open('loo_samples.csv', 'a', newline='')
            test_samples = []
            for loo in range(0,30):
                test_samples.append(['loo',str(loo)])
                torch_fix_seed(seed=loo)
                perm_pos = np.random.permutation(len(pos_list))
                perm_neg = np.random.permutation(len(neg_list))
                p_idxs = perm_pos[num_train_pos+num_val_pos:]
                n_idxs = perm_neg[num_train_neg+num_val_neg:]
                test_samples += [[pos_list[_][0]] for _ in p_idxs]
                test_samples += [[neg_list[_][0]] for _ in n_idxs]
            writer = csv.writer(f0)
            writer.writerows(test_samples)
            f0.close()
        else:
            result_cv = [['best_dice', 'best_auc0', 'epoch_at_best', 'train_auc0_atbest']]
            result_cv_test = [['test_auc0', 'num0']]
            result_cv_test_pred1 = [['lab1', 'pred1']]
            footer = ('adam' if args.adam else 'sgd') + ('_woagesexsmoking' if args.woagesexsmoking else '')
            f0 = open('result_{}.csv'.format(footer), 'a', newline='')
            f1 = open('result_test_{}.csv'.format(footer), 'a', newline='')
            f2 = open('result_test_pred0_{}.csv'.format(footer), 'a', newline='')
            writer = csv.writer(f0)
            writer.writerows(result_cv)
            writer = csv.writer(f1)
            writer.writerows(result_cv_test)
            writer = csv.writer(f2)
            writer.writerows(result_cv_test_pred1)
            f0.close()
            f1.close()
            f2.close()
            for loo in range(0,30):
                torch_fix_seed(seed=loo)
                perm_pos = np.random.permutation(len(pos_list))
                perm_neg = np.random.permutation(len(neg_list))
                train_auc0_atbest = 0
                trial = 0
                while train_auc0_atbest<0.75:
                    trial += 1
                    print('loo {}, trial {}'.format(loo,trial))
                    best_dice, best_auc0, epoch_at_best, train_auc0_atbest, test_auc0, num0, lab0, pred0 = train(args,perm_pos,perm_neg)
                result_cv = [[best_dice, best_auc0, epoch_at_best, train_auc0_atbest]]
                result_cv_test = [[test_auc0, num0]]
                
                preds = [['loo',str(loo)]]
                preds += [[l,p] for l,p in zip(lab0,pred0)]
    
                f0 = open('result_{}.csv'.format(footer), 'a', newline='')
                f1 = open('result_test_{}.csv'.format(footer), 'a', newline='')
                f2 = open('result_test_pred0_{}.csv'.format(footer), 'a', newline='')
                writer = csv.writer(f0)
                writer.writerows(result_cv)
                writer = csv.writer(f1)
                writer.writerows(result_cv_test)
                writer = csv.writer(f2)
                writer.writerows(preds)
                f0.close()
                f1.close()
                f2.close()
        
if __name__ == '__main__':
    main()
