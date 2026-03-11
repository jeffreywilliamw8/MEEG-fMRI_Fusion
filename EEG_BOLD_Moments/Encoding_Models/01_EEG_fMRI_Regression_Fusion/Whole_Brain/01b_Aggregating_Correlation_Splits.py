import numpy as np
import os
import time
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Aggregate Parallelized fMRI Correlations")
    parser.add_argument('--eeg_frequency', type=int, default=100)
    parser.add_argument('--channel_policy', type=str, choices=['ga', 'app'], default='ga')
    parser.add_argument('--num_subjects', type=int, default=10)
    return parser.parse_args()

def aggregate_correlations():
    args = get_args()
    start_time = time.time()
    
    # Constants for fsaverage processing
    num_splits = 21
    v_per_split = 7802
    n_timepoints = 370
    total_vertices = 163842 

    # Base path following the structure of the regression script
    base_path = f'/scratch/jeffreykatab/Code/Encoding_Models/correlations/policy-{args.channel_policy}/{args.eeg_frequency}_Hz'

    subjects = [f"{i:02d}" for i in range(1, args.num_subjects + 1)]

    for sub in tqdm(subjects, desc="Aggregating Subjects"):
        sub_dir = os.path.join(base_path, f"fmri_sub-{sub}")
        
        for hemi in ['left', 'right']:
            hemi_path = os.path.join(sub_dir, f"{hemi}_hemi")
            
            if not os.path.isdir(hemi_path):
                continue

            # Pre-allocate full array (Time, Vertices) for the whole brain surface
            # This is significantly faster than list.extend()
            full_data = np.zeros((n_timepoints, total_vertices), dtype=np.float32)

            for i in range(1, num_splits + 1):
                split_file = os.path.join(hemi_path, f"split-{i:02d}.npy")
                
                if os.path.exists(split_file):
                    split_data = np.load(split_file) # Shape: (370, 7802)
                    
                    # Calculate vertex indices for this split
                    start_v = v_per_split * (i - 1)
                    end_v = start_v + split_data.shape
                    
                    # Stitch the split into the master array
                    full_data[:, start_v:end_v] = split_data
                else:
                    print(f"Warning: Missing split {i} for sub {sub} {hemi}")

            # Save aggregated results to the subject-level directory
            save_path = os.path.join(sub_dir, f"correlations_{hemi}.npy")
            np.save(save_path, full_data)

    end_time = time.time()
    print(f"\nAggregation complete! Total time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    aggregate_correlations()