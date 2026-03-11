import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = ['01', '02', '03']
meg_metadata_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/MEG/prepared'
meg_metadata_file = os.path.join(meg_metadata_dir, 'meg_P1_metadata.npy')
meg_metadata = np.load(meg_metadata_file, allow_pickle=True).item()
fmri_metadata_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/prepared'

times = 1000 * meg_metadata['meg']['times']

PLOTS_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Define Area groupings
areas = [
    ['V1'], 
    ['V2'], 
    ['hV4'], 
    ['IT']  
]
area_labels = ['V1', 'V2', 'hV4', 'IT']

# 2. Storage for final stats
v_means, v_stds = [], []
l_means, l_stds = [], []

print(f"Aggregating Visual vs Language results (NC > 20%) for {len(areas)} areas...")

for area_rois in tqdm(areas, desc="Areas"):
    subject_v_curves = [] 
    subject_l_curves = []
    
    for subject in subject_list:
        # Load metadata for the current subject to access ROI indices and Noise Ceilings
        fmri_metadata = np.load(os.path.join(fmri_metadata_dir, f'fmri_{subject}_metadata.npy'), allow_pickle=True).item()
        whole_brain_nc = fmri_metadata['encoding_model']['noise_ceiling_testset']
        
        roi_v_list = []
        roi_l_list = []
        
        for roi in area_rois:
            # Load the raw ROI timecourses (Time x Voxels) 
            vis_path = f'/scratch/jeffreykatab/Code/Encoding_Models/THINGS/correlations/decoding_encoding/roi/visual/fmri_sub-{subject}/{roi}.npy'
            lang_path = f'/scratch/jeffreykatab/Code/Encoding_Models/THINGS/correlations/decoding_encoding/roi/language/fmri_sub-{subject}/{roi}.npy'
            
            v_data = np.load(vis_path)
            l_data = np.load(lang_path)
            
            # Use metadata to filter voxels based on Noise Ceiling
            # roi_idx maps the ROI voxels to the whole-brain NC array
            roi_idx = fmri_metadata['roi'][roi]
            roi_noise_ceilings = whole_brain_nc[roi_idx]
            
            # Find indices where NC > 20.0
            nc_mask = roi_noise_ceilings > 20.0
            
            # Apply filtering and average across valid voxels
            if np.any(nc_mask):
                v_data_filtered = v_data[:, nc_mask]
                l_data_filtered = l_data[:, nc_mask]
                
                roi_v_list.append(np.mean(v_data_filtered, axis=1))
                roi_l_list.append(np.mean(l_data_filtered, axis=1))
            else:
                print(f"Warning: No voxels passed NC threshold for Sub {subject}, ROI {roi}")

        # Average ROIs for this subject (if data exists)
        if roi_v_list:
            subject_v_curves.append(np.mean(roi_v_list, axis=0))
            subject_l_curves.append(np.mean(roi_l_list, axis=0))

    # Compute stats across subjects
    v_means.append(np.mean(subject_v_curves, axis=0))
    v_stds.append(np.std(subject_v_curves, axis=0))
    
    l_means.append(np.mean(subject_l_curves, axis=0))
    l_stds.append(np.std(subject_l_curves, axis=0))

# --- Plotting (same logic as Code 1) ---
fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
colors = ['#2c7bb6', '#fdae61'] 
vars_labels = ['Visual Features', 'Language Features']
lines = [] 

for i, ax in enumerate(axes):
    lv, = ax.plot(times, v_means[i], color=colors[0], lw=2, label=vars_labels[0])
    ax.fill_between(times, v_means[i] - v_stds[i], v_means[i] + v_stds[i], 
                    color=colors[0], alpha=0.15, linewidth=0)
    
    ll, = ax.plot(times, l_means[i], color=colors[1], lw=2, label=vars_labels[1])
    ax.fill_between(times, l_means[i] - l_stds[i], l_means[i] + l_stds[i], 
                    color=colors[1], alpha=0.15, linewidth=0)
    
    if i == 0: lines = [lv, ll]
    
    ax.set_title(f'Area: {area_labels[i]}', loc='left', fontweight='bold', fontsize=11)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, lw=1)
    ax.axhline(y=0, color='black', alpha=0.2, lw=0.8)
    ax.set_ylabel("Pearson's r")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.legend(lines, vars_labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.96), frameon=False, fontsize=12)
plt.suptitle('Decoding-Encoding: Visual vs. Language', fontsize=16, y=0.99, fontweight='bold')
axes[-1].set_xlabel('Time (ms)')
plt.tight_layout(rect=[0, 0, 1, 0.94])

save_path = os.path.join(PLOTS_DIR, 'DE_Visual_vs_Language.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Execution complete! Plot saved to: {save_path}")
print(f"Execution time: {time.time() - start_time:.2f} seconds.")