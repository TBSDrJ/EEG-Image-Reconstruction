import os
import sys
import numpy as np
# import h5py
import scipy.io as spio
# import nibabel as nib

import torch
import torchvision
import torchvision.models as tvmodels
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import clip

import skimage.io as sio
from skimage import data, img_as_float
from skimage.transform import resize as imresize
from skimage.metrics import structural_similarity as ssim
import scipy as sp

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [0,1,2,5,7]


# images_dir = 'data/nsddata_stimuli/test_images'
# feats_dir = 'data/eval_features/test_images'

# if sub in [1,2,5,7]:
    # feats_dir = 'data/eval_features/subj{:02d}'.format(sub)
    # images_dir = 'results/versatile_diffusion/subj{:02d}'.format(sub)

# images_dir = 'data/things-eeg2_preproc/test_images_direct'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/test_images'

# images_dir = 'results/thingseeg2_preproc/versatile_diffusion'
# feats_dir = 'cache/things-eeg2_preproc/eval_features/subj1_preproc'

# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_200ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_200ms'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_400ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_400ms'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_600ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_600ms'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms'

images_dir = 'results/thingseeg2_preproc/sub10/versatile_diffusion_800ms'
feats_dir = 'cache/thingseeg2_preproc/eval_features/subj10_preproc_800ms'

# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_200ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_200ms_1regressor'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_400ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_400ms_1regressor'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_600ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_600ms_1regressor'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_1regressor'

# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_null'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_null'


# images_dir = 'results/thingseeg2/versatile_diffusion_avg1_200ms'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_200ms'
# images_dir = 'results/thingseeg2/versatile_diffusion_avg1_400ms'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_400ms'
# images_dir = 'results/thingseeg2/versatile_diffusion_avg1_600ms'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_600ms'
# images_dir = 'results/thingseeg2/versatile_diffusion_avg1_800ms'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_800ms'


# images_dir = 'cache/thingsmeg_stimuli/avg_test_images1b'
# feats_dir = 'cache/thingsmeg/eval_features/test_images'

# images_dir = 'results/avg_versatile_diffusion1balltokens/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/subj1'

# images_dir = 'results/thingsmeg/avg_versatile_diffusion_200ms/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_200ms/BIGMEG1'
# images_dir = 'results/thingsmeg/avg_versatile_diffusion_400ms/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_400ms/BIGMEG1'
# images_dir = 'results/thingsmeg/avg_versatile_diffusion_600ms/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_600ms/BIGMEG1'
# images_dir = 'results/thingsmeg/avg_versatile_diffusion_800ms/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_800ms/BIGMEG1'

# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_cliptextonly'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_cliptextonly'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_clipvisiononly'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_clipvisiononly'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_noautokl'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_noautokl'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_nocliptext'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_nocliptext'
# images_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_noclipvision'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_noclipvision'


if not os.path.exists(feats_dir):
   os.makedirs(feats_dir)

class batch_generator_external_images(Dataset):

    def __init__(self, data_path ='', prefix='', net_name='clip'):
        self.data_path = data_path
        self.prefix = prefix
        self.net_name = net_name
        
        if self.net_name == 'clip':
           self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        else:
           self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_test = 200 # 982
        
    def __getitem__(self,idx):
        img = Image.open('{}/{}{}.png'.format(self.data_path,self.prefix,idx))
        img = T.functional.resize(img,(224,224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return  self.num_test





global feat_list
feat_list = []
def fn(module, inputs, outputs):
    feat_list.append(outputs.cpu().numpy())

net_list = [
    ('inceptionv3','avgpool'),
    ('clip','final'),
    ('alexnet',2),
    ('alexnet',5),
    ('efficientnet','avgpool'),
    ('swav','avgpool')
    ]

device = 1
net = None
batchsize=64



for (net_name,layer) in net_list:
    feat_list = []
    print(net_name,layer)
    dataset = batch_generator_external_images(data_path=images_dir,net_name=net_name,prefix='')
    loader = DataLoader(dataset,batchsize,shuffle=False)
    
    if net_name == 'inceptionv3': # SD Brain uses this
        net = tvmodels.inception_v3(pretrained=True)
        if layer== 'avgpool':
            net.avgpool.register_forward_hook(fn) 
        elif layer == 'lastconv':
            net.Mixed_7c.register_forward_hook(fn)
            
    elif net_name == 'alexnet':
        net = tvmodels.alexnet(pretrained=True)
        if layer==2:
            net.features[4].register_forward_hook(fn)
        elif layer==5:
            net.features[11].register_forward_hook(fn)
        elif layer==7:
            net.classifier[5].register_forward_hook(fn)
            
    elif net_name == 'clip':
        model, _ = clip.load("ViT-L/14", device='cuda:{}'.format(device))
        net = model.visual
        net = net.to(torch.float32)
        if layer==7:
            net.transformer.resblocks[7].register_forward_hook(fn)
        elif layer==12:
            net.transformer.resblocks[12].register_forward_hook(fn)
        elif layer=='final':
            net.register_forward_hook(fn)
    
    elif net_name == 'efficientnet':
        net = tvmodels.efficientnet_b1(weights=True)
        net.avgpool.register_forward_hook(fn) 
        
    elif net_name == 'swav':
        net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        net.avgpool.register_forward_hook(fn) 
    net.eval()
    net.cuda(device)    
    
    with torch.no_grad():
        for i,x in enumerate(loader):
            print(i*batchsize)
            x = x.cuda(device)
            _ = net(x)
    if net_name == 'clip':
        if layer == 7 or layer == 12:
            feat_list = np.concatenate(feat_list,axis=1).transpose((1,0,2))
        else:
            feat_list = np.concatenate(feat_list)
    else:   
        feat_list = np.concatenate(feat_list)
    
    
    file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
    np.save(file_name,feat_list)
