import numpy as np
import os
import argparse
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def set_seed(seed=8):
    np.random.seed(seed)
    random.seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description="EEG Data Preprocessing Pipeline")
    parser.add_argument('--eeg_frequency', type=int, default=100)
    parser.add_argument('--subjects', type=int, default=6, help="Number of subjects to process")
    return parser.parse_args()

def average_repeats(data, is_train=True):
    """
    Averages EEG responses across repeats and sessions.
    Train stimuli: 250 (indices m to m+250)
    Test stimuli: 102 (indices 1001 to 1102)
    """
    session_averages = []
    
    for session_idx in range(8):
        presentation_order = data['stimuli_presentation_order'][session_idx]
        eeg_data = data['eeg_data'][session_idx]
        
        if is_train:
            m = min(np.unique(presentation_order))
            stim_range = range(m, m + 250)
        else:
            stim_range = range(1001, 1103)
            
        stim_averages = []
        for stim_id in stim_range:
            indices = np.where(presentation_order == stim_id)
            stim_averages.append(np.mean(eeg_data[indices], axis=0))
        session_averages.append(stim_averages)
    
    session_averages = np.array(session_averages) # (Sessions, Stims, Chans, Time)
    
    if is_train:
        # Custom logic for training: average pairs of sessions (0,4), (1,5), etc.
        paired_avg = []
        for i in range(4):
            paired_avg.append(np.mean([session_averages[i], session_averages[i+4]], axis=0))
        return np.concatenate(paired_avg, axis=0) # (1000, Chans, Time)
    else:
        # Standard average across all sessions for test data
        return np.mean(session_averages, axis=0) # (102, Chans, Time)

def apply_scaling_and_pca(data_list, n_components=127):
    """
    Applies Z-scoring and PCA (at each time point) across a list of subject data.
    """
    z_scored_list = []
    pca_list = []
    
    for eeg in data_list:
        n_stims, n_chans, n_times = eeg.shape
        z_out = np.zeros_like(eeg)
        p_out = np.zeros((n_stims, n_components, n_times))
        
        for t in range(n_times):
            # Z-Score
            scaler = StandardScaler()
            z_out[:, :, t] = scaler.fit_transform(eeg[:, :, t])
            
            # PCA
            pca = PCA(n_components=n_components, random_state=8)
            p_out[:, :, t] = pca.fit_transform(z_out[:, :, t])
            
        z_scored_list.append(z_out)
        pca_list.append(p_out)
        
    return z_scored_list, pca_list

def main():
    args = get_args()
    set_seed()
    
    base_path = "/scratch/jeffreykatab/eeg_moments/dataset/preprocessed_data/eeg/sub-{:02d}/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-{:04d}"
    output_dir = f"/scratch/jeffreykatab/Code/preprocessed_eeg/{args.eeg_frequency}_Hz/unified_outputs"
    os.makedirs(output_dir, exist_ok=True)

    train_raw_list = []
    test_raw_list = []

    # 1. Load and Average Repetitions
    for sub_id in range(1, args.subjects + 1):
        print(f"Loading Subject {sub_id}...")
        path = base_path.format(sub_id, args.eeg_frequency)
        data = np.load(os.path.join(path, 'preprocessed_data.npy'), allow_pickle=True).item()
        
        train_raw_list.append(average_repeats(data, is_train=True))
        test_raw_list.append(average_repeats(data, is_train=False))

    # 2. Process Transformations (Z-Score and PCA)
    print("Processing Training Data...")
    train_z, train_pca = apply_scaling_and_pca(train_raw_list)
    print("Processing Test Data...")
    test_z, test_pca = apply_scaling_and_pca(test_raw_list)

    # 3. Create Aggregated Versions
    # --- Averaged across subjects ---
    train_z_avg = np.mean(train_z, axis=0)
    train_pca_avg = np.mean(train_pca, axis=0)
    test_z_avg = np.mean(test_z, axis=0)
    test_pca_avg = np.mean(test_pca, axis=0)

    # --- Appended across channel dimension ---
    train_z_app = np.concatenate(train_z, axis=1)
    train_pca_app = np.concatenate(train_pca, axis=1)
    test_z_app = np.concatenate(test_z, axis=1)
    test_pca_app = np.concatenate(test_pca, axis=1)

    # 4. Save Everything
    outputs = {
        "train_ss_z": {f"sub-{i+1:02d}": d for i, d in enumerate(train_z)},
        "train_ss_pca": {f"sub-{i+1:02d}": d for i, d in enumerate(train_pca)},
        "test_ss_z": {f"sub-{i+1:02d}": d for i, d in enumerate(test_z)},
        "test_ss_pca": {f"sub-{i+1:02d}": d for i, d in enumerate(test_pca)},
        
        "train_group_avg_z": train_z_avg,
        "train_group_avg_pca": train_pca_avg,
        "test_group_avg_z": test_z_avg,
        "test_group_avg_pca": test_pca_avg,
        
        "train_group_app_z": train_z_app,
        "train_group_app_pca": train_pca_app,
        "test_group_app_z": test_z_app,
        "test_group_app_pca": test_pca_app
    }

    for name, data in outputs.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), data)
    
    print(f"Successfully saved all outputs to {output_dir}")

if __name__ == "__main__":
    main()