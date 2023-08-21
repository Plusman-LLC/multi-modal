#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
# import unet_resblock3_focus2
import cv2
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
import pathlib
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from scipy import interp
from scipy import stats
import csv
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'
RAW_DATA = True
auc_flag=True
    
class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)
    
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, s in enumerate(slices):
        axes[i].imshow(s, cmap="gray", origin="lower")

img_all = []
aucs=[]
fpr_list=[]
tpr_list=[]
metrics=[]
classifier=[]

pred_result_filename = './result_test_pred0_adam.csv'
column1 = ['loo','label','pred']
df = pd.DataFrame(data = None, columns = column1)            
with open(pred_result_filename) as f:
    reader = csv.reader(f)
    loo = ''
    for row in reader:
        if row[0] == 'loo':
            loo = int(row[1])
        elif loo!='':
            df = pd.concat([df,pd.DataFrame(data = [[loo,int(row[0]),float(row[1])]], columns = column1)])          

# fig = plt.figure(figsize=(8, 8), dpi=192)
fig = plt.figure(figsize=(10, 10), dpi=192)
trial = 1
for l in range(loo+1):
    SUVs = []
    lab = []
    df_ = df[df['loo']==l]
    for index, row in df_.iterrows():
        print(row)
        lab.append(row['label'])
        SUVs.append(row['pred'])
    if len(lab)>0:
        #ROC
        fpr, tpr, thresholds = roc_curve(np.array(lab),np.array(SUVs))
        auc0 = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        aucs.append(auc0)
        print(auc0)
        #Classification
        inter = tpr-fpr
        idx = np.argmax(inter)    
        classification = (np.array(SUVs)>=thresholds[idx]).astype(np.int8)
        plt.plot(fpr, tpr, marker='o', linestyle = "dashed", label='Trial_{} (auc = %.2f)'.format(trial)%auc0)
        plt.legend()
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()
        trial += 1

aucs=np.array(aucs)
result = np.mean(aucs)
print('AUC: '+str(result))

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_list]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(fpr_list)):
    mean_tpr += interp(all_fpr, fpr_list[i], tpr_list[i])

# Finally average it and compute AUC
mean_tpr /= len(fpr_list)

plt.plot(all_fpr, mean_tpr, marker='o', label='All (auc = %.2f)'%result)
plt.legend()
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()

plt.savefig('ROC_25trial.png')
