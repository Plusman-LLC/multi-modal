import os, sys
import pathlib
import glob
import shutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
import pickle
import skimage
import skimage.morphology
from PIL import Image
import scipy
from scipy import ndimage
from skimage import measure, morphology, filters

root_dir = 'path to working dir'

pat_data_list = [
['4501415323',62,1,1],
['0629280e80',69,1,1],
['d32bdf57b4',61,1,1],
['c954ef01a1',77,0,1],
['fe01fa0c75',61,0,1],
['b3e6842e73',58,1,1],
['03db138356',55,0,1],
['ec18549ed8',62,0,1],
['44210c6d75',56,1,1],
['cb57f4ba3d',65,1,1],
['30b71225fb',62,1,1],
['3c4b48e0c8',44,0,1],
['c78f411492',79,0,1],
['1979445574',50,1,1],
['e97f1f3cf2',47,0,1],
['55d48eacc8',70,0,1],
['2282ddafb8',61,0,1],
['801d1ad55d',40,0,1],
['51b47309fa',47,0,1],
['0f41cf6d92',68,0,1],
['bb967f330c',64,0,1],
['8a46872327',61,1,1],
['25762b9857',85,1,1],
['ccec18f4a0',74,0,1],
['9a81b8d0f3',36,1,1],
['484f43ad34',44,0,1],
['5d5b1ebf63',73,1,1],
['0bbf62e564',59,1,1],
['a54f50bbf9',63,0,1],
['f5da1752dd',55,0,1],
['0a675fb1e7',40,0,0],
['d603d74324',56,0,0],
['c264582866',41,0,0],
['c7abf08685',72,0,0],
['e523926aad',70,0,0],
['643b188a61',83,0,0],
['b863ac2008',68,1,0],
['d841d4d51b',62,0,0],
['dcc82c8430',84,0,0],
['6c83932a75',66,0,0],
['bd938ca201',67,1,0],
['f503ea1b41',68,1,0],
['9e1341e188',69,0,0],
['42e40bde8e',53,0,0],
['0491f5a9e2',72,0,0],
['63e911de8c',47,0,0],
['be024405c5',54,0,0],
['c3c80b2eae',85,0,0],
['f8872691a8',60,0,0],
['156935e6bc',80,1,0],
['c146bb9dc3',69,0,0],
['5bae7867df',60,1,0],
['4783c31eb1',76,0,0],
['cedb508b3e',56,0,0],
['157eb46a5d',82,1,0],
['1a285c8cad',61,1,0],
['b836979521',55,0,0],
['fa6d62116f',72,0,0],
['825b9139e7',57,0,0],
['e4b366ba45',62,0,0],
['25e11fecf4',65,0,0],
['9a2b4fb54c',84,1,0]    
    
    ]

pat_list = [_[0] for _ in pat_data_list]

def area_to_contour(area, color):
    kernel = np.ones((3,3), np.uint8)
    area_dilute = cv2.dilate(area,kernel,iterations = 1)
    contour = area_dilute - area
    contour0 = np.zeros(contour.shape, np.uint8)
    if color==1:
        contour_rgb = np.dstack([contour, contour0, contour0])
    elif color==2:
        contour_rgb = np.dstack([contour0, contour, contour0])
    else:
        contour_rgb = np.dstack([contour0, contour0, contour])
    contour_mask = np.asarray((contour>0), dtype=np.uint8)
    c = np.dstack([contour_mask, contour_mask, contour_mask])
    return contour_rgb, c

def detect_mark_on_image(img, mask1=None, mask2=None, mask3=None):
    zeros = np.zeros((*img.shape, 3)).astype(np.uint8)
    contour_t, ct = area_to_contour(mask1, 1) if mask1 is not None else (zeros, zeros)
    contour_y, cy = area_to_contour(mask2, 2) if mask2 is not None else (zeros, zeros)
    contour_y2, cy2 = area_to_contour(mask3, 0) if mask3 is not None else (zeros, zeros)
        
    img_rgb = np.dstack([img, img, img])
    
    # Merge image and contour
#    img_rgb = img_rgb*(1-ct)*(1-cy) + 255*contour_y*(1-ct) + 255*contour_t
    img_rgb = img_rgb*(1-ct)*(1-cy)*(1-cy2) + 255*contour_y2*(1-ct)*(1-cy) + + 255*contour_y*(1-ct) + 255*contour_t
    return img_rgb

def crop(x, min_HU, max_HU, uint8=True):
    return_array = (np.clip(x, min_HU, max_HU) - min_HU) / (max_HU - min_HU)
    if uint8:return_array = (return_array*255).astype(np.uint8)
    return return_array

def toimage(hu,WL,WW,mask=None):
    low = WL-WW/2
    img = (np.clip(hu-low,0,WW)/WW*255).astype("u1")
    img = np.stack([img,img,img],2)
    if mask is not None: img[mask==1] = np.array([255,0,0])
    return Image.fromarray(img)
def contour(mask):
    return skimage.morphology.binary_dilation(mask.astype("i")) - mask.astype("i")
def get_foreground(image):
    region = skimage.measure.regionprops((image>0).astype("i"))[0]
    return region.bbox
def blur(image, blur_size):
    image = [skimage.filters.gaussian(_im, blur_size) for _im in image]
    return np.array(image)
def mask_body(hu):
    labels = skimage.measure.label(hu>-200)
    regions = skimage.measure.regionprops(labels)
    regions.sort(key=lambda x:-x.area)
    #Image.fromarray(255*(hu>-100).astype("u1")[199])
    masks = np.array([skimage.morphology.convex_hull_image(label==regions[0].label) for label in labels])
    return np.where(masks>0, hu, hu.min())

def find_all_dirs(directory):
    for root, dirs, files in os.walk(directory):
        try:
            if len([s for s in files if 'Segmentation' in s]):
#            if os.path.exists(root + '/*.nrrd'):
                yield root
        except:
            pass
        
if True:
    out_dir = root_dir + '/train_data'
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    
    dirs = [_ for _ in find_all_dirs(root_dir + '/data/3dslicer')]
    dirs += [_ for _ in find_all_dirs(root_dir + '/data/3dslicer_pos/training 202303')]
    
    patiens = [os.path.basename(d) for d in dirs]

    step=1
    stats = []
    error_data = []
    sample_image_list = ['c3c80b2eae']#出力したい患者番号
    for d in dirs[:]:
        pat = os.path.basename(d)
        pat_index = pat_list.index(pat)
        pat_data = pat_data_list[pat_index]
        try:
            files = glob.glob(d+'/Segmentation*.seg.nrrd')
            path_mask = files[0]
            path_image = d + '/*Helical*.nrrd'
            path_image = glob.glob(path_image)[0]

            mask = sitk.ReadImage(path_mask)
            spacing = mask.GetSpacing()
            spx, spy, spz = spacing
            mask = sitk.GetArrayFromImage(mask).astype('f')#ndarrayを取得
            
            num_labels = [np.sum(mask==_) for _ in range(1)]
            
            if spz != 2.0:
                print('not 2.0')
#                sys.stop
            print(pat,spz,spx,np.sum(mask>-100),num_labels)
            # stats.append(np.hstack([np.array([patname,spz,spx,np.sum(mask>-100)]),num_labels]))

            zoom = [spz,spy,spx]/np.array([1.0,1.0,1.0]) # (2,1.5,1.5)mmに変換

            hu = sitk.ReadImage(path_image)
            hu = sitk.GetArrayFromImage(hu).astype('f')#ndarrayを取得

            hu = scipy.ndimage.zoom(hu, zoom, order=1)
            mask = scipy.ndimage.zoom(mask, zoom, order=0).astype("i")

            image = hu.astype("i2")
            mask = mask.astype("i1")

            y_label = measure.label(mask, background=0)
            regions = measure.regionprops(y_label)
            if len(regions)>1:
                areas = [_.area for _ in regions]
                areas2 = sorted(areas,reverse=True)
                idx = areas.index(areas2[0])
                region = regions[idx]
            else:
                region = regions[0]
            iso = 50 #mm
            
            z_margin = iso//2#(iso-(region.bbox[3]-region.bbox[0]))/2
            y_margin = iso//2#(iso-(region.bbox[4]-region.bbox[1]))/2
            x_margin = iso//2#(iso-(region.bbox[5]-region.bbox[2]))/2
            
            center_voxel = [(region.bbox[3]+region.bbox[0])//2,(region.bbox[4]+region.bbox[1])//2,(region.bbox[5]+region.bbox[2])//2]
            
            nz_min, nz_max = max(0,center_voxel[0]-int(z_margin)),center_voxel[0]+int(z_margin)
            ny_min = center_voxel[1]-y_margin
            ny_max = center_voxel[1]+y_margin
            nx_min = center_voxel[2]-x_margin
            nx_max = center_voxel[2]+x_margin
            
            nodule_crop = hu[nz_min:nz_max,ny_min:ny_max,nx_min:nx_max]
            mask_crop = mask[nz_min:nz_max,ny_min:ny_max,nx_min:nx_max]>0
            
            token = np.array([pat_data[1],pat_data[2]],dtype=np.float32)
            label = pat_data[3]
            # nodule_crops.append(nodule_crop)
            # mask_crops.append(mask_crop)
            # tokens.append(token)
                            
            # if len(nodule_crops)>0:
            o_path = out_dir \
                    + '/' + str(label) \
                    + '/' + str(pat) 
            os.makedirs(o_path,exist_ok=True)
            with open(o_path+'/nodule_crop.pkl', 'wb') as f:
                pickle.dump(nodule_crop, f)
            with open(o_path+'/mask_crop.pkl', 'wb') as f:
                pickle.dump(mask_crop, f)
            with open(o_path+'/token.pkl', 'wb') as f:
                pickle.dump(token, f)
            step+=1
        except Exception as e:
            print(e)
    # stats=np.stack(stats)