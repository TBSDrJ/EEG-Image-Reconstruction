import sys
import os
import numpy as np
import sklearn.linear_model as skl
import argparse
import pickle
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

# nsd_features = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub))
# train_latents = nsd_features['train_latents']
# test_latents = nsd_features['test_latents']
train_latents = np.load('cache/thingseeg2_preproc/extracted_embeddings/train_autokl.npy', mmap_mode='r')
test_latents = np.load('cache/thingseeg2_preproc/extracted_embeddings/test_autokl.npy', mmap_mode='r')

# train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
# train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_assumehrf_sub{}.npy'.format(sub,sub)
# train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_brainmask_sub{}.npy'.format(sub,sub)
# train_fmri = np.load(train_path)
# test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
# test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_assumehrf_sub{}.npy'.format(sub,sub)
# test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_brainmask_sub{}.npy'.format(sub,sub)
# test_fmri = np.load(test_path)
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg_200ms.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg_400ms.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg_600ms.npy'
train_path = 'data/things-eeg2_preproc/sub10/train_thingseeg2_avg_800ms.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_null.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_200ms.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_400ms.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_600ms.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_800ms.npy'
train_meg = np.load(train_path, mmap_mode='r')
# train_meg = train_meg[:8000,:,:]
train_meg = train_meg.reshape(train_meg.shape[0], -1)
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg_200ms.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg_400ms.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg_600ms.npy'
test_path = 'data/things-eeg2_preproc/sub10/test_thingseeg2_avg_800ms.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_null.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_200ms.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_400ms.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_600ms.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_800ms.npy'
test_meg = np.load(test_path, mmap_mode='r')
# test_meg = test_meg[:1000,:,:]
test_meg = test_meg.reshape(test_meg.shape[0], -1)
print(train_meg.shape, test_meg.shape)

## Preprocessing fMRI

train_fmri = train_meg
test_fmri = test_meg


norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print(np.mean(train_fmri),np.std(train_fmri))
print(np.mean(test_fmri),np.std(test_fmri))

print(np.max(train_fmri),np.min(train_fmri))
print(np.max(test_fmri),np.min(test_fmri))

# -3.1423596606535043e-16 0.9999435586284325
# -0.027261797786477318 0.9914779354685911
# 24.4805317560972 -14.412738287655525
# 9.81827287708914 -9.315348672150476

# brainmask
# -9.90978801924559e-17 0.9999435586284213
# 0.00021609721579497063 0.9921138744865566
# 40.97194429840731 -44.864775508707616
# 24.379927104507665 -26.19067915181277

num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)

## latents Features Regression
print('Training latents Feature Regression')

reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.fit(train_fmri, train_latents)
print('Training complete')


pred_test_latent = reg.predict(test_fmri)
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
print(reg.score(test_fmri,test_latents))
# -0.00520250264941022
# -0.005210581349707694 null
# -0.005233597313805604 avg1_200ms
# -0.0053340178161875385 avg1_400ms

# np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_latents)
# np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_assumehrf_pred_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_latents)
# np.save('data/predicted_features/subj{:02d}/nsd_vdvae_brainmask_pred_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_latents)
# subject = 'BIGMEG1'
# save_dir = 'cache/predicted_embeddings/' + subject + '/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + f'thingsmeg_regress_autokl_sub-{subject}.npy', pred_latents)

# save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl.npy', pred_latents)

# save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_200ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_400ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_600ms.npy', pred_latents)

save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/sub10/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(save_dir + 'thingseeg2_regress_autokl_800ms.npy', pred_latents)


# save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_null.npy', pred_latents)


# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_avg1_200ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_avg1_400ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_avg1_600ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_avg1_800ms.npy', pred_latents)


del train_fmri
datadict = {
    'weight' : reg.coef_,
    'bias' : reg.intercept_,

}

# with open('data/regression_weights/subj{:02d}/vdvae_regression_weights.pkl'.format(sub),"wb") as f:
# with open('data/regression_weights/subj{:02d}/vdvae_regression_weights_assumehrf.pkl'.format(sub),"wb") as f:
# # with open('data/regression_weights/subj{:02d}/vdvae_brainmask_regression_weights.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)

# subject = 'BIGMEG1'
# save_dir = 'cache/regression_weights/' + subject + '/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + f'thingsmeg_regress_autokl_weights_sub-{subject}.pkl', "wb") as f:
#     pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2_preproc/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights.npy', pred_latents)

# save_dir = 'cache/thingseeg2_preproc/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_200ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2_preproc/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_400ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2_preproc/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_600ms.npy', pred_latents)

save_dir = 'cache/thingseeg2_preproc/regression_weights/sub10/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + 'thingseeg2_regress_autokl_weights_800ms.pkl', "wb") as f:
    pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2_preproc/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_null.npy', pred_latents)


# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_avg1_200ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_avg1_400ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_avg1_600ms.npy', pred_latents)

# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_autokl_weights_avg1_800ms.npy', pred_latents)