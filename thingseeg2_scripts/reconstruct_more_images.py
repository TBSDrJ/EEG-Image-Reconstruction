# %%
import numpy as np
import pickle
import sklearn.linear_model as skl
import os
from scipy.spatial.distance import correlation
from tqdm import tqdm

gpu1 = 6
gpu2 = 7

# epochs = np.load('cache/thingseeg2_synthetic/sub01_4to0__5_1__800ms.npy')
epochs = np.load('cache/thingseeg2_synthetic/sub01_ica16_800ms.npy')
# epochs = np.load('cache/thingseeg2_synthetic/not_much_real__cat_real__0_80_800ms.npy')
# epochs = np.load('data/things-eeg2_preproc/test_thingseeg2_avg_800ms.npy')

epochs = epochs.reshape(epochs.shape[0], -1) # concatenate all time points
# vdvae_recon_dir = 'results/thingseeg2_synthetic/vdvae_not_much_real__cat_real__0_20_800ms/'
# vdvae_recon_dir = 'results/thingseeg2_synthetic/vdvae_sub01_4to0__5_1__800ms.npy/'
vdvae_recon_dir = 'results/thingseeg2_synthetic/vdvae_sub01_ica16_800ms.npy/'
# vdvae_recon_dir = 'results/thingseeg2_synthetic/vdvae_temp/'
# diffusion_recon_dir = 'results/thingseeg2_synthetic/versatile_diffusion_not_much_real__cat_real__0_20_800ms/'
# diffusion_recon_dir = 'results/thingseeg2_synthetic/versatile_diffusion_sub01_4to0__5_1__800ms/'
diffusion_recon_dir = 'results/thingseeg2_synthetic/versatile_diffusion_sub01_ica16_800ms/'
# diffusion_recon_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_noautokl/'
# diffusion_recon_dir = 'results/thingseeg2_preproc/versatile_diffusion_800ms_clipvisiononly/'
# diffusion_recon_dir = 'results/thingseeg2_synthetic/versatile_diffusion_temp/'

# %% [markdown]
### Predict VDVAE Features

# %%
with open('cache/thingseeg2_preproc/regression_weights/thingseeg2_regress_autokl_weights_800ms.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

# %%
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.coef_ = reg_w
reg.intercept_ = reg_b

train_latents = np.load('cache/thingseeg2_preproc/extracted_embeddings/train_autokl.npy', mmap_mode='r')
# pred_test_latents = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_autokl_800ms.npy', mmap_mode='r')
mean_pred_test_latent = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_autoklmean_800ms.npy', mmap_mode='r')
std_pred_test_latent = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_autoklstd_800ms.npy', mmap_mode='r')

# %%
pred_test_latent = reg.predict(epochs)
# std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
# np.save('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_autoklmean_800ms.npy', np.mean(pred_test_latent,axis=0))
# np.save('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_autoklstd_800ms.npy', np.std(pred_test_latent,axis=0))
std_norm_test_latent = (pred_test_latent - mean_pred_test_latent) / std_pred_test_latent
pred_autokl = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
# pred_autokl = pred_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
# pred_autokl = pred_test_latent

# %%
# save_dir = 'cache/thingseeg2_synthetic/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_not_much_real__cat_real__20_40_800ms.npy', pred_autokl)

# %% [markdown]
### Predict CLIP-Text Features
# %% 
with open('cache/thingseeg2_preproc/regression_weights/thingseeg2_regress_cliptext_weights_800ms.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

# %%

train_clip = np.load('cache/thingseeg2_preproc/extracted_embeddings/train_cliptext.npy', mmap_mode='r')
# pred_test_latents = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_800ms.npy', mmap_mode='r')
mean_pred_test_latent = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptextmean_800ms.npy', mmap_mode='r')
std_pred_test_latent = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptextstd_800ms.npy', mmap_mode='r')

# %%
num_embed = 77
pred_cliptext = np.zeros((epochs.shape[0], num_embed, 768))
# mean_pred_test_latent = np.zeros((num_embed, 768))
# std_pred_test_latent = np.zeros((num_embed, 768))
for i in tqdm(range(num_embed), total=num_embed):
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True) # old alpha=100000, optimal alpha=1200000

    reg.coef_ = reg_w[i]
    reg.intercept_ = reg_b[i]
    
    pred_test_latent = reg.predict(epochs)
    # std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    # mean_pred_test_latent[i] = np.mean(pred_test_latent,axis=0)
    # std_pred_test_latent[i] = np.std(pred_test_latent,axis=0)
    # std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latents[:,i],axis=0)) / np.std(pred_test_latents[:,i],axis=0)
    std_norm_test_latent = (pred_test_latent - mean_pred_test_latent[i]) / std_pred_test_latent[i]
    pred_cliptext[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    # pred_cliptext[:,i] = pred_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    # pred_cliptext[:,i] = pred_test_latent

    # # Compute the Euclidean distances
    # euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # # Compute the average Euclidean distance
    # average_euclidean_distance = euclidean_distances.mean()
    # correlations = (1 - correlation_distances).mean()

    # print(i,reg.score(averaged_epochs,test_clip[:,i]), average_euclidean_distance, correlations)
# np.save('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptextmean_800ms.npy', mean_pred_test_latent)
# np.save('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptextstd_800ms.npy', std_pred_test_latent)

# %%
# save_dir = 'cache/thingseeg2_synthetic/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_cliptext_not_much_real__cat_real__20_40_800ms.npy', pred_cliptext)

# %% [markdown]
### Predict CLIP-Vision Features

# %%
with open('cache/thingseeg2_preproc/regression_weights/thingseeg2_regress_clipvision_weights_800ms.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

train_clip = np.load('cache/thingseeg2_preproc/extracted_embeddings/train_clipvision.npy', mmap_mode='r')
# pred_test_latents = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_800ms.npy', mmap_mode='r')
mean_pred_test_latent = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvisionmean_800ms.npy', mmap_mode='r')
std_pred_test_latent = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvisionstd_800ms.npy', mmap_mode='r')

# %%
num_embed = 257
pred_clipvision = np.zeros((epochs.shape[0], num_embed, 768))
# mean_pred_test_latent = np.zeros((num_embed, 768))
# std_pred_test_latent = np.zeros((num_embed, 768))
for i in tqdm(range(num_embed), total=num_embed):
    reg = skl.Ridge(alpha=60000, max_iter=50000, fit_intercept=True) # old alpha=60000, optimal alpha=300000
    
    reg.coef_ = reg_w[i]
    reg.intercept_ = reg_b[i]
    
    pred_test_latent = reg.predict(epochs)
    # std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    # mean_pred_test_latent[i] = np.mean(pred_test_latent,axis=0)
    # std_pred_test_latent[i] = np.std(pred_test_latent,axis=0)
    # std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latents[:,i],axis=0)) / np.std(pred_test_latents[:,i],axis=0)
    std_norm_test_latent = (pred_test_latent - mean_pred_test_latent[i]) / std_pred_test_latent[i]
    pred_clipvision[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    # pred_clipvision[:,i] = pred_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    # pred_clipvision[:,i] = pred_test_latent

    # # Compute the Euclidean distances
    # euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # # Compute the average Euclidean distance
    # average_euclidean_distance = euclidean_distances.mean()
    # correlations = (1 - correlation_distances).mean()
    
    # print(i,reg.score(test_fmri,test_clip[:,i]), average_euclidean_distance, correlations)
# np.save('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvisionmean_800ms.npy', mean_pred_test_latent)
# np.save('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvisionstd_800ms.npy', std_pred_test_latent)

# %%
# save_dir = 'cache/thingseeg2_synthetic/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision_not_much_real__cat_real__20_40_800ms.npy', pred_clipvision)

# %% [markdown]
### Reconstruct VDVAE images

# %%
import sys
sys.path.append('vdvae')
import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=30)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
batch_size=int(args.bs)

print('Libs imported')

H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Models is Loading')
ema_vae = load_vaes(H)

  
class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)


    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        #img = img/255
        #img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)


# image_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
image_path = 'data/things-eeg2_preproc/test_images.npy'
test_images = batch_generator_external_images(data_path = image_path)
testloader = DataLoader(test_images,batch_size,shuffle=False)

# test_latents = []
for i,x in enumerate(testloader):
  data_input, target = preprocess_fn(x)
  with torch.no_grad():
        print(i*batch_size)
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        #recons = ema_vae.decoder.out_net.sample(px_z)
        # batch_latent = []
        # for i in range(31):
        #     batch_latent.append(stats[i]['z'].cpu().numpy().reshape(len(data_input),-1))
        # test_latents.append(np.hstack(batch_latent))
        #stats_all.append(stats)
        #imshow(imgrid(recons, cols=batch_size,pad=20))
        #imshow(imgrid(test_images[i*batch_size : (i+1)*batch_size], cols=batch_size,pad=20))
# test_latents = np.concatenate(test_latents)      


# pred_latents = np.load(save_dir + 'thingseeg2_regress_autokl_not_much_real__cat_real__20_40_800ms.npy')
pred_latents = pred_autokl.copy()

ref_latent = stats

# Transfor latents from flattened representation to hierarchical
def latent_transformation(latents, ref):
  layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
  transformed_latents = []
  for i in range(31):
    t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
    #std_norm_test_latent = (t_lat - np.mean(t_lat,axis=0)) / np.std(t_lat,axis=0)
    #renorm_test_latent = std_norm_test_latent * np.std(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0) + np.mean(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0)
    c,h,w=ref[i]['z'].shape[1:]
    transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
  return transformed_latents

idx = range(len(pred_latents))
input_latent = latent_transformation(pred_latents[idx],ref_latent)

  
def sample_from_hier_latents(latents,sample_ids):
  sample_ids = [id for id in sample_ids if id<len(latents[0])]
  layers_num=len(latents)
  sample_latents = []
  for i in range(layers_num):
    sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
  return sample_latents

#samples = []

for i in range(int(np.ceil(len(test_images)/batch_size))):
  print(i*batch_size)
  samp = sample_from_hier_latents(input_latent,range(i*batch_size,(i+1)*batch_size))
  px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
  sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
  upsampled_images = []
  for j in range(len(sample_from_latent)):
      im = sample_from_latent[j]
      im = Image.fromarray(im)
      im = im.resize((512,512),resample=3)

      if not os.path.exists(vdvae_recon_dir):
          os.makedirs(vdvae_recon_dir)
      im.save(vdvae_recon_dir + '{}.png'.format(i*batch_size+j))
      

# %% [markdown]
### Reconstruct images

# %%
import sys
sys.path.append('versatile_diffusion')
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-diff_str", "--diff_str",help="Diffusion Strength",default=0.75)
parser.add_argument("-mix_str", "--mix_str",help="Mixing Strength",default=0.4)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
strength = 0.75 # 0.75 normal, 0.99 max
mixing = 0.4 # 0 for pure clipvision, 1 for pure cliptext


def regularize_image(x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x

cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    


# Might require editing the GPU assignments due to Memory issues
net.clip.cuda(gpu1)
net.autokl.cuda(gpu1)

#net.model.cuda(1)
sampler = sampler(net)
#sampler.model.model.cuda(1)
#sampler.model.cuda(1)
batch_size = 1

# save_dir = 'cache/thingseeg2_synthetic/predicted_embeddings/'
# pred_cliptext = np.load(save_dir + 'thingseeg2_regress_cliptext_not_much_real__cat_real__20_40_800ms.npy')
pred_cliptext = torch.tensor(pred_cliptext).half().cuda(gpu2)

# pred_clipvision = np.load(save_dir + 'thingseeg2_regress_clipvision_not_much_real__cat_real__20_40_800ms.npy')
pred_clipvision = torch.tensor(pred_clipvision).half().cuda(gpu2)


n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'
net.autokl.half()

torch.manual_seed(0)
for im_id in range(len(pred_clipvision)):

    zim = Image.open(vdvae_recon_dir + '{}.png'.format(im_id))
    # zim = Image.new('RGB', (64, 64), (128, 128, 128))
   
    zim = regularize_image(zim)
    zin = zim*2 - 1
    zin = zin.unsqueeze(0).cuda(gpu1).half()

    init_latent = net.autokl_encode(zin)
    
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    #strength=0.75
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    device = 'cuda:' + str(gpu1)
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
    #z_enc,_ = sampler.encode(init_latent.cuda(1).half(), c.cuda(1).half(), torch.tensor([t_enc]).to(sampler.model.model.diffusion_model.device))

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.cuda(gpu2).half()
    
    dummy = torch.zeros((1,3,224,224)).cuda(gpu1)
    uim = net.clip_encode_vision(dummy)
    uim = uim.cuda(gpu2).half()
    
    z_enc = z_enc.cuda(gpu2)

    h, w = 512,512
    shape = [n_samples, 4, h//8, w//8]

    cim = pred_clipvision[im_id].unsqueeze(0)
    ctx = pred_cliptext[im_id].unsqueeze(0)
    
    #c[:,0] = u[:,0]
    #z_enc = z_enc.cuda(1).half()
    
    sampler.model.model.diffusion_model.device='cuda:' + str(gpu2)
    sampler.model.model.diffusion_model.half().cuda(gpu2)
    #mixing = 0.4
    
    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        # first_conditioning=[uim, uim],
        second_conditioning=[utx, ctx],
        # second_conditioning=[utx, utx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image', 
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1-mixing), )
    
    z = z.cuda(gpu1).half()
    x = net.autokl_decode(z)
    color_adj='None'
    #color_adj_to = cin[0]
    color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
    color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
    color_adj_keep_ratio = 0.5
    
    if color_adj_flag and (ctype=='vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
    else:
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]
    

    if not osp.exists(diffusion_recon_dir):
        os.makedirs(diffusion_recon_dir)
    x[0].save(diffusion_recon_dir + '{}.png'.format(im_id))
        
        
      

