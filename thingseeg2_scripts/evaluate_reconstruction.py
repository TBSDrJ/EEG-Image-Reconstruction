import os
import sys
import numpy as np
# import h5py
import scipy.io as spio
# import nibabel as nib
import scipy as sp
from PIL import Image



import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]


from scipy.stats import pearsonr,binom,linregress
import numpy as np
def pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)#cosine_similarity(ground_truth, predictions)#
    r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
    #print(r.shape)
    # congruent pairs are on diagonal
    congruents = np.diag(r)
    #print(congruents)
    
    # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
    success = r < congruents
    success_cnt = np.sum(success, 0)
    
    # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
    perf = np.mean(success_cnt) / (len(ground_truth)-1)
    p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
    
    return perf, p


net_list = [
    ('inceptionv3','avgpool'),
    ('clip','final'),
    ('alexnet',2),
    ('alexnet',5),
    ('efficientnet','avgpool'),
    ('swav','avgpool')
    ]

# feats_dir = 'data/eval_features/subj{:02d}'.format(sub)
# test_dir = 'data/eval_features/test_images'
# num_test = 982

num_test = 200
test_dir = 'cache/thingseeg2_preproc/eval_features/test_images'
# feats_dir = 'cache/things-eeg2_preproc/eval_features/subj1_preproc'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_200ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_400ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_600ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj10_preproc_800ms'
feats_dir = 'cache/thingseeg2_synthetic/eval_features/subj1_ica16_800ms'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_cliptextonly'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_clipvisiononly'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_noautokl'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_nocliptext'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_noclipvision'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_200ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_400ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_600ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_800ms_1regressor'
# feats_dir = 'cache/thingseeg2_preproc/eval_features/subj1_preproc_null'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_200ms'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_400ms'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_600ms'
# feats_dir = 'cache/thingseeg2/eval_features/subj1_avg1_800ms'
# test_dir = 'cache/thingsmeg/eval_features/test_images'
# feats_dir = 'cache/thingsmeg/eval_features/subj1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_200ms/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_400ms/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_600ms/BIGMEG1'
# feats_dir = 'cache/thingsmeg/eval_features/avg_versatile_diffusion_800ms/BIGMEG1'


distance_fn = sp.spatial.distance.correlation
pairwise_corrs = []
for (net_name,layer) in net_list:
    file_name = '{}/{}_{}.npy'.format(test_dir,net_name,layer)
    gt_feat = np.load(file_name)
    
    file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
    eval_feat = np.load(file_name)
    
    gt_feat = gt_feat.reshape((len(gt_feat),-1))
    eval_feat = eval_feat.reshape((len(eval_feat),-1))
    
    print(net_name,layer)
    if net_name in ['efficientnet','swav']:
        print('distance: ',np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)]).mean())
    else:
        pairwise_corrs.append(pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test])[0])
        print('pairwise corr: ',pairwise_corrs[-1])
        
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
        
ssim_list = []
pixcorr_list = []
# for i in range(982):
#     gen_image = Image.open('results/versatile_diffusion/subj{:02d}/{}.png'.format(sub,i)).resize((425,425))
#     gt_image = Image.open('data/nsddata_stimuli/test_images/{}.png'.format(i))
#     gen_image = np.array(gen_image)/255.0
#     gt_image = np.array(gt_image)/255.0
#     pixcorr_res = np.corrcoef(gt_image.reshape(1,-1), gen_image.reshape(1,-1))[0,1]
#     pixcorr_list.append(pixcorr_res)
#     gen_image = rgb2gray(gen_image)
#     gt_image = rgb2gray(gt_image)
#     ssim_res = ssim(gen_image, gt_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
#     ssim_list.append(ssim_res)

for i in range(200):
    gt_image = Image.open('data/things-eeg2_preproc/test_images_direct/{}.png'.format(i)).resize((512,512)) # either both 512 or both 500
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_200ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_400ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_600ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_800ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/sub10/versatile_diffusion_800ms/{}.png'.format(i))
    gen_image = Image.open('results/thingseeg2_synthetic/versatile_diffusion_sub01_ica16_800ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_800ms_cliptextonly/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_800ms_clipvisiononly/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_800ms_noautokl/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_800ms_nocliptext/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_800ms_noclipvision/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_200ms_1regressor/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_400ms_1regressor/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_600ms_1regressor/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_800ms_1regressor/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2_preproc/versatile_diffusion_null/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2/versatile_diffusion_avg1_200ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2/versatile_diffusion_avg1_400ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2/versatile_diffusion_avg1_600ms/{}.png'.format(i))
    # gen_image = Image.open('results/thingseeg2/versatile_diffusion_avg1_800ms/{}.png'.format(i))

    # gt_image = Image.open('cache/thingsmeg_stimuli/avg_test_images1b/{}.png'.format(i)).resize((512,512)) # either both 512 or both 500
    # gen_image = Image.open('results/avg_versatile_diffusion1balltokens/BIGMEG1/{}.png'.format(i))
    # gen_image = Image.open('results/thingsmeg/avg_versatile_diffusion_200ms/BIGMEG1/{}.png'.format(i))
    # gen_image = Image.open('results/thingsmeg/avg_versatile_diffusion_400ms/BIGMEG1/{}.png'.format(i))
    # gen_image = Image.open('results/thingsmeg/avg_versatile_diffusion_600ms/BIGMEG1/{}.png'.format(i))
    # gen_image = Image.open('results/thingsmeg/avg_versatile_diffusion_800ms/BIGMEG1/{}.png'.format(i))

    gen_image = np.array(gen_image)/255.0
    gt_image = np.array(gt_image)/255.0
    pixcorr_res = np.corrcoef(gt_image.reshape(1,-1), gen_image.reshape(1,-1))[0,1]
    pixcorr_list.append(pixcorr_res)
    gen_image = rgb2gray(gen_image)
    gt_image = rgb2gray(gt_image)
    ssim_res = ssim(gen_image, gt_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
    ssim_list.append(ssim_res)
    
ssim_list = np.array(ssim_list)
pixcorr_list = np.array(pixcorr_list)
print('PixCorr: {}'.format(pixcorr_list.mean()))
print('SSIM: {}'.format(ssim_list.mean()))

