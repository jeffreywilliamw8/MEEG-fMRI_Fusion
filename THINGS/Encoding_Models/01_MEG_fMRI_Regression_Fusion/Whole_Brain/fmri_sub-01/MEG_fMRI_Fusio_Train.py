import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV
import time

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print(f'>>> Whole-Brain MEG-fMRI Fusion (Training) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the in vivo THINGS fMRI1 train responses
# =============================================================================
# Load the fMRI responses
fmri_dir = os.path.join('/home/jeffreykatab/Projects/fusion/THINGS', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    'fmri_sub-01_split-train.h5')
fmri_train = h5py.File(fmri_dir, 'r')['neural_data']

if 1<=args.fmri_split<=532: # Subject 01 has 211,339 voxels. We split them into 533 splits (532 splits of 397 voxels and 1 split of 135 voxels)
    fmri_train = fmri_train[:, 397*(args.fmri_split - 1):397*args.fmri_split] # Each split has 397 voxels
else:
    fmri_train = fmri_train[:, 211204:] # Last split has 135 voxels to make a total of 211339 voxels

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=1
    )

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


# =============================================================================
# Train the encoding fusion models
# =============================================================================
reg_param = {}
reg_param['coef_'] = []
reg_param['intercept_'] = []
reg_param['alpha_'] = []
reg_param['n_features_in_'] = []

# Loop across MEG time points
alphas = np.logspace(-6, 10, 17)

print("Shape of the MEG train data: ", meg_train.shape)
print("Shape of the fMRI train data: ", fmri_train.shape)

print("Starting training")
for t in tqdm(range(len(times))):

    # Train the encoding fusion models
    reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    reg.fit(meg_train[:,:,t], fmri_train)
    # Store the encoding fusion model weights
    reg_param['coef_'].append(reg.coef_.astype(np.float32))
    reg_param['intercept_'].append(reg.intercept_.astype(np.float32))
    reg_param['alpha_'].append(reg.alpha_.astype(np.float32))
    reg_param['n_features_in_'].append(reg.n_features_in_)

print("Training complete!")

# =============================================================================
# Save the encoding fusion model weights
# =============================================================================
# Create the encoding fusion model weight save directory
save_dir_weights = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'invivo_things_meg_fmri_control', 'encoding_fusion_weights',
    'sub-01')
os.makedirs(save_dir_weights, exist_ok=True)

# Save the weights
file_name = f'fmri_split-{args.fmri_split:03d}.npy'
np.save(os.path.join(save_dir_weights, file_name), reg_param)


# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
