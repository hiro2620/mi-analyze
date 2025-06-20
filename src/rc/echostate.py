import csv
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EchoStateNetwork:
    def __init__(self, n_reservoir=100, spectral_radius=0.95, input_scaling=1.0, 
                 leaking_rate=0.3, random_state=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _generate_reservoir(self, n_inputs):
        # Input weights (random)
        self.W_in = np.random.uniform(-1, 1, (self.n_reservoir, n_inputs))
        self.W_in *= self.input_scaling
        
        # Reservoir weights (sparse random matrix)
        self.W = np.random.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))
        # Make sparse (connect only 10% of neurons)
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) > 0.9
        self.W *= mask
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        self.W *= (self.spectral_radius / max_eigenvalue)
        
    def _update_state(self, x, state):
        # Leaky integrator neuron model
        new_state = (1 - self.leaking_rate) * state + \
                   self.leaking_rate * np.tanh(self.W_in @ x + self.W @ state)
        return new_state
    
    def _collect_states(self, inputs):
        n_samples, n_inputs = inputs.shape
        states = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        
        for i in range(n_samples):
            state = self._update_state(inputs[i], state)
            states[i] = state
            
        return states
    
    def fit(self, X, y):
        # X shape: (n_trials, n_channels, n_timepoints)
        n_trials, n_channels, n_timepoints = X.shape
        
        # Initialize reservoir
        self._generate_reservoir(n_channels)
        
        # Collect reservoir states for all trials
        all_states = []
        for trial in range(n_trials):
            # Transpose to get (time, channels)
            trial_data = X[trial].T
            states = self._collect_states(trial_data)
            # Use only the final state as feature (or could use mean/max of all states)
            all_states.append(states[-1])  # Final reservoir state
            
        all_states = np.array(all_states)
        
        # Train linear readout with ridge regression
        from sklearn.linear_model import Ridge
        self.readout = Ridge(alpha=1e-6)
        self.readout.fit(all_states, y)
        
        return self
    
    def predict(self, X):
        n_trials, n_channels, n_timepoints = X.shape
        
        predictions = []
        for trial in range(n_trials):
            trial_data = X[trial].T
            states = self._collect_states(trial_data)
            final_state = states[-1].reshape(1, -1)
            pred = self.readout.predict(final_state)[0]
            predictions.append(pred)
            
        return np.array(predictions)

def load_eeg_data(file_path):
    """Load and preprocess EEG data following the CSP notebook approach"""
    
    # Load raw EEG data
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    raw.set_montage('easycap-M1')
    raw.filter(l_freq=1, h_freq=40, verbose=False)
    
    # Load task sequences  
    seq_file_dir = Path("/home/neuron/Documents/mi-analyze/data") / "20250604" / "hand3"
    seq_file_paths = [
        seq_file_dir / "task-sequence1.csv",
        seq_file_dir / "task-sequence2.csv", 
        seq_file_dir / "task-sequence3.csv",
    ]
    
    task_sequence = []
    for seq_file_path in seq_file_paths:
        with open(seq_file_path, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if len(row) >= 2:
                    try:
                        task_sequence.append(int(row[1]))
                    except (ValueError, IndexError):
                        print(f"Invalid data in row: {row}")
                        continue
    
    # Extract events
    raw_events, _labels = mne.events_from_annotations(raw, verbose=False)
    triggers_mask = raw_events[:, 2] == 14
    event_timestamps = raw_events[triggers_mask, 0]
    
    # Remove specific problematic timestamps
    ignore_timestamps = [1086766, 1087277, 1087786, 1088297, 1088809, 
                        1090859, 1091369, 1092137, 1092650, 1093161]
    event_timestamps = [t for t in event_timestamps if t not in ignore_timestamps]
    
    assert len(task_sequence) == len(event_timestamps), \
        "Task sequence and event timestamps length mismatch"
    
    # Create events array
    events = []
    for i, t in enumerate(event_timestamps):
        events.append([t, 0, task_sequence[i]])
    events = np.array(events, dtype=int)
    
    # Create epochs
    epochs = mne.Epochs(raw, events, tmin=-1, tmax=4, preload=True, verbose=False)
    
    # Downsample from 2500Hz to 100Hz using averaging
    current_sfreq = epochs.info['sfreq']
    target_sfreq = 100.0
    decimation_factor = int(current_sfreq / target_sfreq)
    
    print(f"Downsampling from {current_sfreq}Hz to {target_sfreq}Hz (factor: {decimation_factor})")
    epochs_resampled = epochs.copy().resample(target_sfreq, verbose=False)
    
    # Get data and labels
    epochs_data = epochs_resampled.get_data(copy=True)
    labels = epochs_resampled.events[:, -1]
    
    return epochs_data, labels, epochs_resampled

def main():
    """Main function to run the Echo State Network classifier"""
    
    print("Loading EEG data...")
    file_path = "/home/neuron/Documents/mi-analyze/data/20250604/20250604_1_ishii.vhdr"
    epochs_data, labels, epochs = load_eeg_data(file_path)
    
    print(f"Data shape: {epochs_data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Crop data to focus on motor imagery period (0-2 seconds)
    epochs_train = epochs.copy().crop(tmin=0.0, tmax=3.0)
    X = epochs_train.get_data(copy=True).astype(np.float64)
    y = labels
    
    print(f"Training data shape: {X.shape}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(X_train.shape, y_train.shape)
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Initialize and train Echo State Network
    print("Training Echo State Network...")
    esn = EchoStateNetwork(
        n_reservoir=5000,
        spectral_radius=0.95,
        input_scaling=1.0,
        leaking_rate=0.3,
        random_state=42
    )
    
    esn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = esn.predict(X_test)
    
    # Convert predictions to discrete labels
    y_pred_labels = np.round(y_pred).astype(int)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Chance level: {max(np.mean(y_test == 1), np.mean(y_test == 2)):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_labels))
    
    # Cross-validation with manual implementation
    print("\nPerforming cross-validation...")
    from sklearn.model_selection import KFold
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1}/5...")
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        esn_cv = EchoStateNetwork(
            n_reservoir=5000,
            spectral_radius=0.95,
            input_scaling=1.0,
            leaking_rate=0.3,
            random_state=42
        )
        
        esn_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = esn_cv.predict(X_val_cv)
        y_pred_cv_labels = np.round(y_pred_cv).astype(int)
        
        acc = accuracy_score(y_val_cv, y_pred_cv_labels)
        cv_scores.append(acc)
        print(f"  Fold {fold + 1} accuracy: {acc:.4f}")
    
    cv_scores = np.array(cv_scores)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")

if __name__ == "__main__":
    main()