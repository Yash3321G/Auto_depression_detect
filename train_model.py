import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def add_noise(data, noise_factor=0.005):
    """Add random white noise to the audio signal for Data Augmentation."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def extract_features(file_name, max_length=200, augment=False):
    """
    Extract MFCC features from an audio file and pad/truncate to a fixed length.
    If 'augment' is True, returns a tuple: (original_features, augmented_features)
    """
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_name, sr=None)
        
        # Helper function to process a single audio array
        def process_audio(y):
            # Extract 40 MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40)
            mfccs = mfccs.T  # Transpose to (time_steps, 40)
            
            # Pad or truncate
            if len(mfccs) > max_length:
                mfccs = mfccs[:max_length, :]
            else:
                pad_width = max_length - len(mfccs)
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            return mfccs
            
        original_features = process_audio(audio)
        
        if augment:
            # Create a noisy version of the audio and extract features
            noisy_audio = add_noise(audio)
            augmented_features = process_audio(noisy_audio)
            return original_features, augmented_features
            
        return original_features
        
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(f"Error details: {e}")
        return (None, None) if augment else None

def main():
    # 1. Configuration Settings
    dataset_path = 'dataset.csv'
    audio_dir = 'audio'
    model_save_path = 'cnn_audio_model.keras'
    max_time_steps = 200
    n_mfcc = 40

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please make sure the file exists.")
        return

    # 2. Load dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Process audio files and extract features with Augmentation
    print("Extracting features (with Data Augmentation)... (this might take a while)")
    features = []
    labels = []

    for index, row in train_df.iterrows():
        file_path = os.path.join(audio_dir, str(row['audio']))
        label = row['label']

        if not os.path.exists(file_path):
            continue

        result = extract_features(file_path, max_length=max_time_steps, augment=True)

        if result[0] is not None:
            orig_feat, aug_feat = result

            features.append(orig_feat)
            labels.append(label)
 
            features.append(aug_feat)  # augmentation ONLY here
            labels.append(label)

    X_test_list = []
    y_test_list = []

    for index, row in test_df.iterrows():
        file_path = os.path.join(audio_dir, str(row['audio']))
        label = row['label']

        if not os.path.exists(file_path):
            continue

        feat = extract_features(file_path, max_length=max_time_steps, augment=False)

        if feat is not None:
            X_test_list.append(feat)
            y_test_list.append(label)
    
    if not features:
        print("Error: No features extracted.")
        return
        
    # 3. Data Processing Pipeline
    X_train = np.array(features)
    y_train = np.array(labels)

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    print(f"Train shape before normalization: {X_train.shape}")
    print(f"Test shape before normalization: {X_test.shape}")

    # Normalize using ONLY train data
    samples, time_steps, n_features = X_train.shape

    scaler = StandardScaler()

    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_scaled.reshape(samples, time_steps, n_features)

    # Apply SAME scaler to test
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test = X_test_scaled.reshape(X_test.shape[0], time_steps, n_features)

    print(f"Normalized train shape: {X_train.shape}")
    print(f"Normalized test shape: {X_test.shape}")
    
    # Calculate Class Weights to handle imbalanced datasets automatically
    class_weights = compute_class_weight(
       class_weight='balanced',
       classes=np.unique(y_train),
       y=y_train
    )
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Calculated Class Weights: {weight_dict}")
    
    
    
    
    # 4. Model Architecture Construction
    print("Building improved 1D CNN model...")
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(max_time_steps, n_mfcc)))
    model.add(BatchNormalization()) # Normalizes layer inputs, stabilizing and speeding up training
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    
    # Fully Connected Layers with heavy Dropout
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) 
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # 5. Training Setup
    print("Compiling model...")
    # Use a slightly lower initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Callbacks
    # 1. EarlyStopping (stop if validation loss doesn't improve for 15 epochs)
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    # 2. ReduceLROnPlateau (reduce learning rate progressively when learning stagnates)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    
    print("Training the CNN model...")
    history = model.fit(
        X_train, y_train, 
        epochs=60, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        class_weight=weight_dict, # Apply weights to balance the dataset
        callbacks=[early_stop, reduce_lr]
    )
    
    # 6. Evaluation
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "=" * 40)
    print(f"Final Model Accuracy on Test Data: {accuracy * 100:.2f}%")
    print("=" * 40)
    
    print("\nTraining History Summary:")
    print(f"Best Training Accuracy: {max(history.history['accuracy']) * 100:.2f}%")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")
    
    # 7. Save model
    print(f"\nSaving improved model to '{model_save_path}'...")
    model.save(model_save_path)
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()
