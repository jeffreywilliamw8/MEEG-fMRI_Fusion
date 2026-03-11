import numpy as np
import os
import pickle
import gc
import random
import argparse
import time
import tqdm
from sklearn.linear_model import RidgeCV

# Reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description="EEG-fMRI Encoding Model (Regression)")
    parser.add_argument('--subject', type=str, default='01', help="fMRI subject ID")
    parser.add_argument('--hemisphere', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--eeg_frequency', type=int, default=100)
    parser.add_argument('--channel_policy', type=str, choices=['ga', 'app'], default='ga', 
                        help="ga: Grand Average subjects, app: Append subjects along channel dim")
    parser.add_argument('--fmri_split', type=int, default=1, 
                        help="Current split index for parallel processing")
    parser.add_argument('--total_splits', type=int, default=21, 
                        help="Total number of splits (163842 / 7802 = 21)")
    return parser.parse_args()

def main():
    args = get_args()
    start_time = time.time()
    
    # 1. Load EEG Data
    # ga -> 'train_ga_pca.npy' | app -> 'train_app_pca.npy'
    eeg_path = f'/scratch/jeffreykatab/Code/preprocessed_eeg/{args.eeg_frequency}_Hz/final_outputs'
    train_file = f'train_{args.channel_policy}_pca.npy'
    test_file = f'test_{args.channel_policy}_pca.npy'
    
    print(f"Loading EEG data ({args.channel_policy}) from {train_file}...")
    eeg_train = np.load(os.path.join(eeg_path, train_file), allow_pickle=True)
    eeg_test = np.load(os.path.join(eeg_path, test_file), allow_pickle=True)

    # 2. Load and Split fMRI Data
    # The fsaverage surface has 163,842 vertices per hemisphere. 
    # To parallelize, we process 7,802 vertices per split.
    V_PER_SPLIT = 7802
    start_v = V_PER_SPLIT * (args.fmri_split - 1)
    end_v = V_PER_SPLIT * args.fmri_split
    
    fmri_dir = f'/scratch/jeffreykatab/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{args.subject}/prepared_betas'
    train_path = os.path.join(fmri_dir, f'sub-{args.subject}_organized_betas_task-train_hemi-{args.hemisphere}_normalized.pkl')
    test_path = os.path.join(fmri_dir, f'sub-{args.subject}_organized_betas_task-test_hemi-{args.hemisphere}_normalized.pkl')

    print(f"Loading fMRI data for vertices {start_v} to {end_v}...")
    with open(train_path, 'rb') as f:
        # We average across repetitions (axis 1) before slicing the split
        fmri_train = np.mean(pickle.load(f), axis=1, dtype=np.float32)[:, start_v:end_v]
    with open(test_path, 'rb') as f:
        fmri_test = np.mean(pickle.load(f), axis=1, dtype=np.float32)[:, start_v:end_v]

    gc.collect()

    # 3. Encoding Model (Ridge Regression)
    # We predict fMRI voxel activity from EEG principal components at each time point.
    alphas = np.logspace(-1, 1, 20)
    n_times = eeg_train.shape
    n_vertices = fmri_train.shape
    correlations = np.zeros((n_times, n_vertices), dtype=np.float32)

    print(f"Starting Regression (Timepoints: {n_times})...")
    for t in tqdm.tqdm(range(n_times)):
        X_train = eeg_train[:, :, t]
        X_test = eeg_test[:, :, t]
        
        # Fit model
        reg = RidgeCV(alphas=alphas)
        reg.fit(X_train, fmri_train)
        
        # Predict and calculate correlation per vertex
        pred_fmri = reg.predict(X_test)
        # Correlation between predicted and actual test BOLD for this vertex
        correlations[t, :] = np.array([np.corrcoef(pred_fmri[:, v], fmri_test[:, v])[0, 1] for v in range(n_vertices)], dtype=np.float32)

    # 4. Save Correlations
    out_dir = f'/scratch/jeffreykatab/Code/Encoding_Models/correlations/policy-{args.channel_policy}/fmri_sub-{args.subject}/{args.hemisphere}_hemi'
    os.makedirs(out_dir, exist_ok=True)
    
    save_name = f'split-{args.fmri_split:02d}.npy'
    np.save(os.path.join(out_dir, save_name), correlations)

    elapsed = time.time() - start_time
    print(f"Finished! Split {args.fmri_split} processed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()