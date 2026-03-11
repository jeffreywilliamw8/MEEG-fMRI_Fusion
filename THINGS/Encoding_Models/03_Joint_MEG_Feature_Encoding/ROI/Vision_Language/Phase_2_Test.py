import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV, LinearRegression
import time
from sklearn.decomposition import PCA
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--modality', default='visual', type=str)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Joint MEG-Feature Encoding Phase 2 (testing) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)

# Start time
start_time = time.time()

# =============================================================================
# Load the in vivo THINGS fMRI1 test responses
# =============================================================================
# Load the fMRI responses
fmri_dir = os.path.join('/home/jeffreykatab/Projects/fusion/THINGS', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    f'fmri_sub-{args.fmri_subject:02d}_split-test.h5')
fmri_test_all = h5py.File(fmri_dir, 'r')['neural_data']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject
    )

# Get the image files names
test_stimuli_fmri = metadata_fmri['encoding_model']['test_stimuli']
unique_test_stimuli = np.unique(test_stimuli_fmri)

# Average the fMRI responses across repetitions of the same test stimulus
fmri_test = []
for stim in tqdm(unique_test_stimuli):
    idx = np.where(test_stimuli_fmri == stim)[0]
    fmri_test.append(fmri_test_all[idx].mean(0))
fmri_test = np.array(fmri_test)

roi_idx = metadata_fmri['roi'][args.roi]
fmri_test = fmri_test[:, roi_idx].astype(np.float32)
print("Shape of the fMRI data: {}".format(fmri_test.shape))

# =============================================================================
# Load and append the in vivo THINGS MEG1 test responses across subjects
# =============================================================================
# Loop across MEG subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Load the MEG responses
    meg_dir = os.path.join('/home/jeffreykatab/Projects/fusion/THINGS', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_test_all = h5py.File(meg_dir, 'r')['neural_data']

    # Time point selection
    tmax = 1.3
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]
    meg_test_all = meg_test_all[:,:,time_idx].astype(np.float32)

    # Average the MEG responses across repetitions for the images shared with
    # the fMRI
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    for stim in unique_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
    meg_test_sub = np.array(meg_test_sub)

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg_test = meg_test_sub
    else:
        meg_test = np.append(meg_test, meg_test_sub, 1)
    del meg_test_all, meg_test_sub

# =============================================================================
# Loading the visual/language features data
# =============================================================================
if args.modality == 'visual':
    features_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/ViT_B_32'
    test_features = np.load(os.path.join(features_dir, 'ViT_B_32_features_768_PCs.npy'), allow_pickle=True).item()['test'][:, :250]
    test_features_stimuli = list(np.load(os.path.join(features_dir, 'ViT_B_32_features_768_PCs.npy'), allow_pickle=True).item()['test_order_map'].keys())

elif args.modality == 'language':
    features_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/language/image_description_embeddings'
    test_features = np.load(os.path.join(features_dir, 'language_features_all-mpnet-base-v2.npy'), allow_pickle=True).item()['pca_test_features'][:, :250] # P.S: 213 PCs explain 95% of the variance
    test_features_stimuli = np.load(os.path.join(features_dir, 'language_features_all-mpnet-base-v2.npy'), allow_pickle=True).item()['test_stimuli_names']


# Ensuring order alignment between the fMRI responses and the stimulus features by sorting the stimulus features based on the image IDs of the fMRI training images

idx_feat = []
for stim in unique_test_stimuli:
    idx_feat.append(test_features_stimuli.index(stim))
idx_feat = np.array(idx_feat)
test_features = test_features[idx_feat,:].astype(np.float32)
print("Shape of the stimulus features (test): {}".format(test_features.shape))


# =============================================================================
# Test the encoding fusion models
# =============================================================================
# Load the encoding fusion model regression weights
weights_file = f'/scratch/jeffreykatab/Code/Encoding_Models/THINGS/regression_weights/joint_meg_feature_encoding/roi/phase_2/{args.modality}/fmri_sub-{args.fmri_subject:02d}/{args.roi}.npy'
reg_param = np.load(weights_file, allow_pickle=True).item()



# Empty correlation array of shape:
# (281 MEG time points, number of voxels in the current fMRI split)
n_voxels = fmri_test.shape[1]
n_time_points = len(times)
correlations = np.zeros((n_time_points, n_voxels), dtype=np.float32)

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  \
    (fmri_test.std(0) + eps)

print("Shape of the MEG test data: ", meg_test.shape)
print("Shape of the fMRI test data: ", fmri_test.shape)

# Loop across MEG time points
print("Starting testing")
for t in tqdm(range(len(times))):

    # Instantiate the fusion regression model
    reg = LinearRegression()
    reg.coef_ = reg_param['coef_'][t]
    reg.intercept_ = reg_param['intercept_'][t]

    # Generate the t-fMRI responses for the test images with in vivo MEG
    tfmri = reg.predict(test_features)

    # Center and normalize the t-fMRI responses
    tfmri_z = (tfmri - tfmri.mean(0)) /  (tfmri.std(0) + eps)

    # Correlate the t-fMRI test responses with the fMRI test responses
    correlations[t,:] = np.diag(tfmri_z.T @ fmri_test_z) / len(tfmri_z)

    # Delete unused variables
    del tfmri, tfmri_z, reg
    gc.collect()
del fmri_test, fmri_test_z
gc.collect()
print("Testing complete!")


# =============================================================================
# Save the encoding fusion model weights and correlation coefficients
# =============================================================================

file_name = f'{args.roi}.npy'

corrs_dir = f'/scratch/jeffreykatab/Code/Encoding_Models/THINGS/correlations/joint_meg_feature_encoding/roi/{args.modality}/fmri_sub-{args.fmri_subject:02d}'

if os.path.isdir(corrs_dir) == False:
    os.makedirs(corrs_dir)
np.save(os.path.join(corrs_dir, file_name), correlations)
print("Saved correlations to: ", os.path.join(corrs_dir, file_name))


# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")