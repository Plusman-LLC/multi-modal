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
from torchvision import transforms
from PIL import Image
from utils import util
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

import monai
from datetime import datetime

from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import radiomics
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, shape2D
import six
import SimpleITK as sitk

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
# import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import precision_recall_fscore_support  as prf
from sklearn.linear_model import LogisticRegression

import logging

#pyradiomics settings
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline
settings['label'] = 1 # 

#radiomics normalize
#ex. joint average
mean = np.array([-4.60686523e+02,  5.57343330e+01,  5.55692480e+08,  4.49466753e+00,
        2.50626862e+02,  1.85487556e+01,  6.63596985e+02,  1.66055496e+02,
       -1.57113998e+02, -1.10246269e+02, -9.27955200e+02,  1.59155225e+03,
        1.10984482e+02,  3.10540955e+02, -5.36212325e-01,  5.55692480e+08,
        7.25209713e-02,  5.43014609e+04,  6.85553551e-01,  5.37778020e-01,
        1.40130882e+01,  2.69212437e+01,  2.88900986e+01,  2.92722912e+01,
        2.38584156e+01,  3.23000221e+01,  7.32495605e+03,  1.78398819e+01,
        7.22207308e-01,  2.24807080e+03,  5.22954643e-01,  7.34955225e+03,
        1.32621228e+03,  1.17411750e+07,  7.23161963e+03,  2.41517380e+02,
        6.53889389e+01,  5.58380365e-01,  4.93282747e+00,  3.62737656e+00,
        3.54399948e+01,  3.43333840e-01,  2.65002370e-01,  9.79735970e-01,
        9.16122854e-01, -1.71774834e-01,  8.12422812e-01,  2.47952014e-01,
        1.42144868e-02,  7.99696541e+00,  6.68044984e-01,
        4.15496081e-02,  6.53239441e+01,  5.15213299e+00,  7.67265701e+01,
        4.31376892e+02,  6.28485084e-02,  9.06819916e+01,  1.29134033e+03,
        1.52764201e+00,  2.07913428e+03,  6.73202332e-03,  5.48119377e-03,
        5.03704071e+00,  5.04130615e+03,  8.34957838e-01,  8.96571934e-01,
        2.41145968e-01,  9.28002179e-01,  1.18709680e+03,  5.26332157e-03,
        6.97770767e+01,  3.89795192e-02,  1.13580605e+02,  1.10792737e+03,
        2.07563794e+03,  3.16545275e+06,  3.68678141e+00,  8.08229297e-03,
        1.01870789e+03,  4.79537159e-01,  7.17911065e-01,  7.93253052e+02,
        6.61915448e-03,  6.42732430e+00,  3.87518078e-01,  2.06209253e+03],
      dtype=np.float32)
stdev = np.array([-4.60686523e+02,  5.57343330e+01,  5.55692480e+08,  4.49466753e+00,
        2.50626862e+02,  1.85487556e+01,  6.63596985e+02,  1.66055496e+02,
       -1.57113998e+02, -1.10246269e+02, -9.27955200e+02,  1.59155225e+03,
        1.10984482e+02,  3.10540955e+02, -5.36212325e-01,  5.55692480e+08,
        7.25209713e-02,  5.43014609e+04,  6.85553551e-01,  5.37778020e-01,
        1.40130882e+01,  2.69212437e+01,  2.88900986e+01,  2.92722912e+01,
        2.38584156e+01,  3.23000221e+01,  7.32495605e+03,  1.78398819e+01,
        7.22207308e-01,  2.24807080e+03,  5.22954643e-01,  7.34955225e+03,
        1.32621228e+03,  1.17411750e+07,  7.23161963e+03,  2.41517380e+02,
        6.53889389e+01,  5.58380365e-01,  4.93282747e+00,  3.62737656e+00,
        3.54399948e+01,  3.43333840e-01,  2.65002370e-01,  9.79735970e-01,
        9.16122854e-01, -1.71774834e-01,  8.12422812e-01,  2.47952014e-01,
        1.42144868e-02,  7.99696541e+00,  6.68044984e-01,
        4.15496081e-02,  6.53239441e+01,  5.15213299e+00,  7.67265701e+01,
        4.31376892e+02,  6.28485084e-02,  9.06819916e+01,  1.29134033e+03,
        1.52764201e+00,  2.07913428e+03,  6.73202332e-03,  5.48119377e-03,
        5.03704071e+00,  5.04130615e+03,  8.34957838e-01,  8.96571934e-01,
        2.41145968e-01,  9.28002179e-01,  1.18709680e+03,  5.26332157e-03,
        6.97770767e+01,  3.89795192e-02,  1.13580605e+02,  1.10792737e+03,
        2.07563794e+03,  3.16545275e+06,  3.68678141e+00,  8.08229297e-03,
        1.01870789e+03,  4.79537159e-01,  7.17911065e-01,  7.93253052e+02,
        6.61915448e-03,  6.42732430e+00,  3.87518078e-01,  2.06209253e+03],
      dtype=np.float32)

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
    def __init__(self,is_train=True,transform=None,data_list=None):
        self.is_train = is_train
        self.images, self.masks, self.labels0, self.tokens = [], [], [], []
        self.agesexsmoking = []
        def read_input(path2,label0):
            nodule_crops = []
            mask_crops = []
            tokens = []
            try:
                with open(path2+'/nodule_crop.pkl', 'rb') as f:
                    img = pickle.load(f)
                    # nodule_crops += [img]
                    img = sitk.GetImageFromArray(img.astype(np.int16))
                with open(path2+'/mask_crop.pkl', 'rb') as f:
                    msk = pickle.load(f)
                    msk = sitk.GetImageFromArray(msk.astype(np.uint8))
                    # mask_crops += [msk]
                with open(path2+'/token.pkl', 'rb') as f:
                    token = [pickle.load(f)]
                    token2 = [t.astype('f')*0 for t in token]
                    tokens += token2
                agesexsmoking = np.array([token[0][0]/100,token[0][1]],dtype=np.float32)

                _radiomics = []
                # for img, msk in zip(imgs,msks):
                firstOrderFeatures = firstorder.RadiomicsFirstOrder(img, msk, **settings)
                firstOrderFeatures.enableAllFeatures()
                results = firstOrderFeatures.execute()
                vec = [val for (key, val) in six.iteritems(results)]
                ke = [['firstOrderFeatures',key] for (key, val) in six.iteritems(results)]
                
                shapeFeatures = shape.RadiomicsShape(img, msk, **settings)
                shapeFeatures.enableAllFeatures()
                results = shapeFeatures.execute()
                vec += [val for (key, val) in six.iteritems(results)]
                ke += [['shapeFeatures',key] for (key, val) in six.iteritems(results)]

                glcmFeatures = glcm.RadiomicsGLCM(img, msk, **settings)
                glcmFeatures.enableAllFeatures()
                results = glcmFeatures.execute()
                # vec += [val for (key, val) in six.iteritems(results)]
                # ke += [['glcmFeatures',key] for (key, val) in six.iteritems(results)]
                vec += [val for (key, val) in six.iteritems(results) if key!='JointAverage']
                ke += [['glcmFeatures',key] for (key, val) in six.iteritems(results) if key!='JointAverage']
                
                glrlmFeatures = glrlm.RadiomicsGLRLM(img, msk, **settings)
                glrlmFeatures.enableAllFeatures()
                results = glrlmFeatures.execute()
                vec += [val for (key, val) in six.iteritems(results)]
                ke += [['glrlmFeatures',key] for (key, val) in six.iteritems(results)]

                glszmFeatures = glszm.RadiomicsGLSZM(img, msk, **settings)
                glszmFeatures.enableAllFeatures()
                results = glszmFeatures.execute()
                vec += [val for (key, val) in six.iteritems(results)]
                ke += [['glszmFeatures',key] for (key, val) in six.iteritems(results)]

                vec = np.array(vec,dtype=np.float32)
                vec = (vec - mean)/stdev
                _radiomics.append(vec)


            except:
                print('error in ')
                return
            
            self.images.append(_radiomics)
            self.tokens.append(tokens)
            self.labels0.append(np.array(label0,dtype=np.int32))
            self.agesexsmoking.append(agesexsmoking)
            self.features = ke
            
        for d in data_list:
            read_input(d[0],d[1])

        print()
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        _images = self.images[index].copy()
        label0 = self.labels0[index].copy()
        _tokens = self.tokens[index].copy()
        agesexsmoking = self.agesexsmoking[index].copy()
        
        if self.transform is not None:
            images, masks, tokens = [],[],[]
            _perm = np.random.permutation(range(len(_images)))
            for p in _perm:
                image, token = _images[p], _tokens[p]
                d = {'image':image[None].astype('f')+1024}
                images.append(d['image'][0]-1024)
                masks.append(d['mask'][0])
                tokens.append(token+np.random.randn(2)*0.05)
        else:
            images = _images
            # masks = _masks
            tokens = _tokens
            
        _data = {'images':images, 'label0':label0, 'tokens':tokens, 'agesexsmoking':agesexsmoking}
        return _data

def count_pos_neg(t):
    num_pos = torch.maximum(torch.IntTensor([1]).cuda(),torch.sum(t>0))
    num_neg = torch.maximum(torch.IntTensor([1]).cuda(),torch.sum(t==0))
    return num_pos[0], num_neg[0]

def count_label(t,num_label):
    temp_list = [torch.maximum(torch.IntTensor([1]).cuda(),torch.sum(t==i)) for i in range(num_label)]
    return torch.cat(temp_list,dim=-1)
def load_data(ld):
    x_trains, x_labels = [], []
    for lab in range(1):
        x_train, x_label = [], []
        for d in ld:
            images = d['images'][0].numpy()
            tokens = d['tokens'][0].numpy()
            agesexsmoking = d['agesexsmoking'].numpy()
            label = d['label{}'.format(lab)].long().numpy()
            if label >= 0: 
                # x_train.append(np.concatenate([images[0],tokens[0],agesexsmoking[0]],axis = 0))
                x_train.append(np.concatenate([images[0],agesexsmoking[0]],axis = 0))
                x_label.append(label)
        x_trains.append(np.stack(x_train))
        x_labels.append(np.stack(x_label))
    return x_trains, x_labels

def train(args, perm_pos, perm_neg):
    epochs = 500
    cycle = 100
    util.set_seeds(args.rank)
    
    data_list = np.array(pos_list)[perm_pos[:num_train_pos]].tolist()
    data_list += np.array(neg_list)[perm_neg[:num_train_neg]].tolist()
    data_list_val = np.array(pos_list)[perm_pos[num_train_pos:num_train_pos+num_val_pos]].tolist()
    data_list_val += np.array(neg_list)[perm_neg[num_train_neg:num_train_neg+num_val_neg]].tolist()
    data_list_test = np.array(pos_list)[perm_pos[num_train_pos+num_val_pos:]].tolist()
    data_list_test += np.array(neg_list)[perm_neg[num_train_neg+num_val_neg:]].tolist()

    train_ds = dataset_lymph(is_train=False,transform=None,data_list=data_list)
    train_ld = monai.data.DataLoader(train_ds, batch_size=1, shuffle=False)
    val_ds = dataset_lymph(is_train=False,transform=None,data_list=data_list_val)
    val_ld = monai.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    test_ds = dataset_lymph(is_train=False,transform=None,data_list=data_list_test)
    test_ld = monai.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    x_trains, y_trains = load_data(train_ld)
    x_vals, y_vals = load_data(val_ld)
    x_tests, y_tests = load_data(test_ld)
    
    depth = None#3
    leaf = 5
    train_aucs, val_aucs, test_aucs = [], [], []
    if args.randomforest:
        for lab in range(1):
            forest = RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=leaf, class_weight='balanced_subsample').fit(x_trains[lab], y_trains[lab])
            score = forest.score(x_tests[lab], y_tests[lab])
            print('score depth{} leaf{}:  {:0.3f}'.format(depth, leaf, score))
            feature_importances_ = forest.feature_importances_
            #train_auc
            try:
                y_pred = forest.predict(x_trains[lab])
                y_score = forest.predict_proba(x_trains[lab])[:,1]
                
                fpr, tpr, thresholds = roc_curve(y_trains[lab], y_score)
                auc0 = auc(fpr, tpr)
                train_aucs.append(auc0)
            except:
                train_aucs.append(np.nan)
    
            #val
            try:
                y_pred_val = forest.predict(x_vals[lab])
                y_score_val = forest.predict_proba(x_vals[lab])[:,1]
                
                fpr, tpr, thresholds = roc_curve(y_vals[lab], y_score_val)
                auc0 = auc(fpr, tpr)
                val_aucs.append(auc0)
            except:
                val_aucs.append(np.nan)
                
            #test
            try:
                y_pred_test = forest.predict(x_tests[lab])
                y_score_test = forest.predict_proba(x_tests[lab])[:,1]
    
                fpr, tpr, thresholds = roc_curve(y_tests[lab], y_score_test)
                auc0 = auc(fpr, tpr)
                test_aucs.append(auc0)
            except:
                test_aucs.append(np.nan)
        return 0, val_aucs[0], 0, train_aucs[0], test_aucs[0], len(x_trains[0]), y_tests[0], y_score_test, train_ds.features, feature_importances_
    else:
        for lab in range(1):
            # lda = LDA(n_components=30)
            pca = PCA(n_components=31)
            lr = LogisticRegression() # ロジスティック回帰モデルのインスタンスを作成
            # lr.fit(X_train, Y_train) # ロジスティック回帰モデルの重みを学習
            steps1 = list(zip(["pca", "gnb"], [pca, lr]))
            # steps2 = list(zip(["lda", "gnb"], [lda, lr]))
            p2 = Pipeline(steps1)
            # p2 = Pipeline(steps2)
            
            p2.fit(x_trains[lab], y_trains[lab])
            
            # forest = RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=leaf, class_weight='balanced_subsample').fit(x_trains[lab], y_trains[lab])
            score = p2.score(x_tests[lab], y_tests[lab])
            print('score depth{} leaf{}:  {:0.3f}'.format(depth, leaf, score))
            
            #train_auc
            try:
                y_pred = p2.predict(x_trains[lab])
                y_score = p2.predict_proba(x_trains[lab])[:,1]
                
                fpr, tpr, thresholds = roc_curve(y_trains[lab], y_score)
                auc0 = auc(fpr, tpr)
                train_aucs.append(auc0)
            except:
                train_aucs.append(np.nan)
    
            #val
            try:
                y_pred_val = p2.predict(x_vals[lab])
                y_score_val = p2.predict_proba(x_vals[lab])[:,1]
                
                fpr, tpr, thresholds = roc_curve(y_vals[lab], y_score_val)
                auc0 = auc(fpr, tpr)
                val_aucs.append(auc0)
            except:
                val_aucs.append(np.nan)
                
            #test
            try:
                y_pred_test = p2.predict(x_tests[lab])
                y_score_test = p2.predict_proba(x_tests[lab])[:,1]
    
                fpr, tpr, thresholds = roc_curve(y_tests[lab], y_score_test)
                auc0 = auc(fpr, tpr)
                test_aucs.append(auc0)
            except:
                test_aucs.append(np.nan)
        
        return 0, val_aucs[0], 0, train_aucs[0], test_aucs[0], len(x_trains[0]), y_tests[0], y_score_test, None, None

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
    # python -m torch.distributed.launch --nproc_per_node=3 main.py --train
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--randomforest', action='store_true') #RMSpropの代わりにAdamとするか
    parser.add_argument('--sgd', action='store_true') #RMSpropの代わりにAdamとするか
    parser.add_argument('--adabelief', action='store_true') #RMSpropの代わりにAdamとするか
    parser.add_argument('--tophat', action='store_true') #tophat bottomhatを追加して5chにするか否か
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--cv', action='store_true')

    args = parser.parse_args()
    args.distributed = False
    args.rank = 0
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.cv:
        result_cv = [['best_dice', 'best_auc0', 'epoch_at_best', 'train_auc0_atbest']]
        result_cv_test = [['test_auc0', 'num0']]
        result_cv_test_pred1 = [['lab1', 'pred1']]
        footer = 'randomforest' if args.randomforest else 'PCA+logistic'
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
            # np.random.seed(seed=loo)
            torch_fix_seed(seed=loo)
            perm_pos = np.random.permutation(len(pos_list))
            perm_neg = np.random.permutation(len(neg_list))
            # perm = np.random.permutation(len(samples_list))
            train_auc0_atbest = 0
            trial = 0
            while train_auc0_atbest<0.75:
                trial += 1
                print('loo {}, trial {}'.format(loo,trial))
                best_dice, best_auc0, epoch_at_best, train_auc0_atbest, test_auc0, num0, lab0, pred0, features, feature_importances_ = train(args,perm_pos,perm_neg)
            
            if args.randomforest:
                if loo == 0:
                    feature_importances = feature_importances_
                else:
                    feature_importances += feature_importances_
            
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
        if args.randomforest:
            features += [['none','age'],['none','gender']]
            for feature, importance in zip (features,feature_importances):
                feature.append(importance/30)
            f0 = open('result_importance_{}.csv'.format(footer), 'a', newline='')
            writer = csv.writer(f0)
            writer.writerows(features)
            f0.close()
        
        
if __name__ == '__main__':
    main()
