import mne
import matplotlib.pyplot as plt

# Load your raw and epochs file
raw = mne.io.read_raw_edf("/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/EDF_files/NU151W.edf", preload=True)
epochs = mne.read_epochs("/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/Epochs/NU151W_epochs-epo.fif", preload=True)

# Convert epoch events into annotations
events = epochs.events
event_id_map = epochs.event_id  # dictionary of {event_name: event_code}

# Make annotations from events
annot_from_events = mne.annotations_from_events(
    events=events,
    event_desc=event_id_map,
    sfreq=raw.info['sfreq']
)

# Add annotations to raw
raw.set_annotations(annot_from_events)

# Plot raw with epoch lines (vertical annotations)
fig = raw.plot(n_channels=50, scalings='auto', show=False)
fig.savefig("epochs_on_raw_plot.png", dpi=300)
print("Saved: epochs_on_raw_plot.png")
