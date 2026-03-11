import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV
import time
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--modality', default='visual', type=str)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Decoding-Encoding Fusion (training) <<<')
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
meg_train = meg_train[4320:,:,:] # Selecting only the MEG responses for the images shared with the fMRI
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
train_features = train_features[4320:,:] # Selecting only the stimulus features for the images shared with the fMRI
print("Shape of the stimulus features (train): {}".format(train_features.shape))

# =============================================================================
# Train the encoding fusion models
# =============================================================================

# Empty dictionary to store the encoding fusion model weights for the ROI
encoder_weights = {}
encoder_weights['coef_'] = []
encoder_weights['intercept_'] = []

decoder_weights = {}
decoder_weights['coef_'] = []
decoder_weights['intercept_'] = []


alphas = np.logspace(-6, 10, 17)

# Loop across MEG time points
print("Starting training")
n_time_points = meg_train.shape[2]
for t in tqdm(range(n_time_points)):

    # Train the decoder
    
    decoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    decoder.fit(meg_train[:,:,t], train_features)
    decoder_weights['coef_'].append(decoder.coef_.astype(np.float32))
    decoder_weights['intercept_'].append(decoder.intercept_.astype(np.float32))
    decoded_features = decoder.predict(meg_train[:,:,t])
    del decoder
    gc.collect()

    # Train the encoder
    
    encoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    encoder.fit(decoded_features, fmri_train)

    # Store the encoding fusion model weights (will be used later for variance partitioning)
    encoder_weights['coef_'].append(encoder.coef_.astype(np.float32))
    encoder_weights['intercept_'].append(encoder.intercept_.astype(np.float32))

print("Training complete!")
# =============================================================================
# Save the encoding fusion model weights
# =============================================================================
file_name = f'{args.roi}.npy'
weights_dir = f'/scratch/jeffreykatab/Code/Encoding_Models/THINGS/regression_weights/decoding_encoding/roi/{args.modality}/fmri_sub-{args.fmri_subject:02d}'
if os.path.isdir(weights_dir) == False:
    os.makedirs(weights_dir)
np.save(os.path.join(weights_dir, 'encoder_weights_'+file_name), encoder_weights)

print("Saved encoder weights to: ", os.path.join(weights_dir, 'encoder_weights_'+file_name))

np.save(os.path.join(weights_dir, 'decoder_weights_'+file_name), decoder_weights)
print("Saved decoder weights to: ", os.path.join(weights_dir, 'decoder_weights_'+file_name))

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")