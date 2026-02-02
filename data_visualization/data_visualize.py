import mne

# Path to the raw GDF file
file_path = "B0101T.gdf"

# Load raw EEG
raw = mne.io.read_raw_gdf(file_path, preload=True)

# Keep EEG channels only (optional but cleaner)
raw.pick_types(eeg=True)

# Visualize the entire raw recording
raw.plot(
    scalings="auto",
    n_channels=16,
    title="B0101T – Raw EEG",
    block=True
)
