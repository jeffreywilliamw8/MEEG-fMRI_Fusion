import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV, LinearRegression
import time


parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--modality', default='visual', type=str)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Joint MEG-Feature Encoding Phase 2 (training) <<<')
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
# Load the in vivo THINGS fMRI1 train responses
# =============================================================================
# Load the fMRI responses
fmri_dir = os.path.join('/home/jeffreykatab/Projects/fusion/THINGS', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    f'fmri_sub-{args.fmri_subject:02d}_split-train.h5')
fmri_train = h5py.File(fmri_dir, 'r')['neural_data']



# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject
    )

roi_idx = metadata_fmri['roi'][args.roi]
fmri_train = fmri_train[4320:, roi_idx].astype(np.float32) # Shape: (4320, n_voxels)
print("Shape of the fMRI data: {}".format(fmri_train.shape))

train_stimuli_fmri = metadata_fmri['encoding_model']['train_stimuli']


# Get the image files names
train_stimuli_fmri = metadata_fmri['encoding_model']['train_stimuli']

# =============================================================================
# Load and append the in vivo THINGS MEG1 train responses across subjects
# =============================================================================
# Loop across subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Time point selection
    tmax = 1.3
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]

    # Load the MEG responses
    meg_train_dir = os.path.join('/home/jeffreykatab/Projects/fusion/THINGS', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_all_training_splits.h5')
    
    # meg_P{msub}_all_training_splits.h5 or meg_P{msub}_split-train.h5
    meg_train_sub = h5py.File(meg_train_dir, 'r')['neural_data']

    # Get the MEG responses for the images shared with the fMRI
    train_stimuli_meg = metadata_meg['encoding_model']['all_training_splits']\
        ['train_stimuli']
    idx_meg = []
    for stim in train_stimuli_fmri:
        idx_meg.append(train_stimuli_meg.index(stim))
    idx_meg = np.array(idx_meg)
    meg_train_sub = meg_train_sub[:,:,time_idx][idx_meg].astype(np.float32)

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg_train = meg_train_sub
    else:
        meg_train = np.append(meg_train, meg_train_sub, 1)
    del meg_train_sub

meg_train = meg_train[4320:,:,:] # Shape: (4320, n_sensors_across_subjects, n_time_points)
print("Shape of the MEG train data: ", meg_train.shape)

# =============================================================================
# Loading the visual/language features data
# =============================================================================
if args.modality == 'visual':
    features_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/ViT_B_32'
    train_features = np.load(os.path.join(features_dir, 'ViT_B_32_features_768_PCs.npy'), allow_pickle=True).item()['train'][:, :250] # Selecting only the first 250 PCs
    train_features_stimuli = list(np.load(os.path.join(features_dir, 'ViT_B_32_features_768_PCs.npy'), allow_pickle=True).item()['train_order_map'].keys())

elif args.modality == 'language':
    features_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/language/image_description_embeddings'
    train_features = np.load(os.path.join(features_dir, 'language_features_all-mpnet-base-v2.npy'), allow_pickle=True).item()['pca_train_features'][:, :250] # Selecting only the first 250 PCs
    train_features_stimuli = np.load(os.path.join(features_dir, 'language_features_all-mpnet-base-v2.npy'), allow_pickle=True).item()['train_stimuli_names']


# Ensuring order alignment between the fMRI responses and the stimulus features by sorting the stimulus features based on the image IDs of the fMRI training images
idx_feat = []
for stim in train_stimuli_fmri:
    idx_feat.append(train_features_stimuli.index(stim))
idx_feat = np.array(idx_feat)
train_features = train_features[idx_feat,:].astype(np.float32)
train_features = train_features[4320:, :]
print("Shape of the stimulus features (train): {}".format(train_features.shape))


# =============================================================================
# Load the MEG-to-fMRI encoder's predictions for the training images
# =============================================================================
weights_path = os.path.join(args.berg_dir, 'jmfe_phase_1', 'roi', f'fmri_sub-{args.fmri_subject:02d}')
#              os.path.join(args.berg_dir, 'jmfe_phase_1', 'roi', f'fmri_sub-{args.fmri_subject:02d}')
phase_1_weights = np.load(os.path.join(weights_path, f'{args.roi}.npy'), allow_pickle=True).item() # Loading the trained models' weights
print("Number of time points in the weights object: ", len(phase_1_weights['coef_']))


# =============================================================================
# Train the encoding fusion models
# =============================================================================

# Empty dictionary to store the encoding fusion model weights for the ROI
reg_param = {}
reg_param['coef_'] = []
reg_param['intercept_'] = []


alphas = np.logspace(-6, 10, 17)

# Loop across MEG time points
print("Starting training")
n_time_points = len(phase_1_weights['coef_'])
for t in tqdm(range(n_time_points)):
    meg2fmri = LinearRegression()
    meg2fmri.coef_ = phase_1_weights['coef_'][t]
    meg2fmri.intercept_ = phase_1_weights['intercept_'][t]
    t_fmri = meg2fmri.predict(meg_train[:,:,t])

    # Train the encoding fusion models
    
    encoding_model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    encoding_model.fit(train_features, t_fmri)

    # Store the encoding fusion model weights (will be used later for variance partitioning)
    reg_param['coef_'].append(encoding_model.coef_.astype(np.float32))
    reg_param['intercept_'].append(encoding_model.intercept_.astype(np.float32))

print("Training complete!")
# =============================================================================
# Save the encoding fusion model weights
# =============================================================================

file_name = f'{args.roi}.npy'
weights_dir = f'/scratch/jeffreykatab/Code/Encoding_Models/THINGS/regression_weights/joint_meg_feature_encoding/roi/phase_2/{args.modality}/fmri_sub-{args.fmri_subject:02d}'
if os.path.isdir(weights_dir) == False:
    os.makedirs(weights_dir)
np.save(os.path.join(weights_dir, file_name), reg_param)

print("Saved weights to: ", os.path.join(weights_dir, file_name))
# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")