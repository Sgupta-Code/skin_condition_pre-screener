import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import gc
import time

# Aggressive memory optimization settings
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)


def build_ultra_lightweight_model(input_shape=(64, 64, 3), num_classes=7):
    """Build an extremely lightweight CNN model for skin condition classification."""
    inputs = Input(shape=input_shape)

    # Dramatically simplified architecture
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    # Minimal classification head
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Use SGD optimizer which can be more efficient on CPU
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_mobilenetv2_minimal(input_shape=(96, 96, 3), num_classes=7):
    """Build an extremely efficient MobileNetV2-based model using alpha=0.35 (smallest version)."""

    # Load the pre-trained model with alpha=0.35 (smallest version)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        alpha=0.35  # Use the smallest version of MobileNetV2
    )

    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create a very simple model with minimal parameters
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')  # Direct connection to output
    ])

    # Use SGD which can be more efficient on CPU
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def process_data_in_chunks(data_dir, img_size, batch_size=8):
    """Process and save data in chunks to avoid loading all into memory."""
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Get the number of samples and classes
    num_samples = generator.samples
    num_classes = generator.num_classes
    class_indices = generator.class_indices

    # Create directory for processed data if it doesn't exist
    processed_dir = os.path.join(os.path.dirname(data_dir), 'processed_chunks')
    os.makedirs(processed_dir, exist_ok=True)

    # Process data in chunks
    chunk_size = 500  # Process 500 images at a time
    for i in range(0, num_samples, chunk_size):
        end_idx = min(i + chunk_size, num_samples)
        print(f"Processing chunk {i // chunk_size + 1}: images {i + 1}-{end_idx}")

        # Get the data for this chunk
        x_chunk = []
        y_chunk = []
        for j in range(i, end_idx, batch_size):
            x_batch, y_batch = next(generator)
            x_chunk.append(x_batch)
            y_chunk.append(y_batch)

        # Concatenate the batches
        x_chunk = np.concatenate(x_chunk)
        y_chunk = np.concatenate(y_chunk)

        # Save the chunk
        np.savez_compressed(
            os.path.join(processed_dir, f'chunk_{i // chunk_size + 1}.npz'),
            x=x_chunk, y=y_chunk
        )

        # Force garbage collection
        del x_chunk, y_chunk
        gc.collect()

    return num_samples, num_classes, class_indices


class DataChunkGenerator:
    """A generator that loads data chunks from disk one at a time."""

    def __init__(self, chunk_dir, batch_size=8, shuffle=True):
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith('.npz')])
        self.current_chunk = None
        self.current_indices = None
        self.chunk_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_chunk is None or len(self.current_indices) < self.batch_size:
            # Load next chunk if needed
            if self.chunk_index >= len(self.chunk_files):
                self.chunk_index = 0
                raise StopIteration

            chunk_path = os.path.join(self.chunk_dir, self.chunk_files[self.chunk_index])
            chunk_data = np.load(chunk_path)
            self.current_chunk = (chunk_data['x'], chunk_data['y'])
            self.chunk_index += 1

            # Create indices for this chunk
            self.current_indices = np.arange(len(self.current_chunk[0]))
            if self.shuffle:
                np.random.shuffle(self.current_indices)

        # Get batch indices
        batch_indices = self.current_indices[:self.batch_size]
        self.current_indices = self.current_indices[self.batch_size:]

        # Return the batch
        return self.current_chunk[0][batch_indices], self.current_chunk[1][batch_indices]


def train_model_efficient():
    """Train skin condition model with extreme memory optimizations for CPU."""

    # Create directories
    os.makedirs('../models', exist_ok=True)

    # Smaller image size for efficiency
    img_width, img_height = 64, 64
    img_size = (img_width, img_height)

    # Very small batch size for CPU
    batch_size = 8

    print("Step 1: Processing data in chunks...")

    # Process the datasets in chunks
    train_samples, num_classes, class_indices = process_data_in_chunks(
        '../data/processed/train', img_size, batch_size
    )
    val_samples, _, _ = process_data_in_chunks(
        '../data/processed/val', img_size, batch_size
    )
    test_samples, _, _ = process_data_in_chunks(
        '../data/processed/test', img_size, batch_size
    )

    # Save class labels
    class_labels = {v: k for k, v in class_indices.items()}
    import json
    with open('../models/class_labels.json', 'w') as f:
        json.dump(class_labels, f)

    print("Step 2: Building ultra-lightweight model...")

    # Build model - choose the simplest model for CPU
    model = build_ultra_lightweight_model(
        input_shape=(img_width, img_height, 3),
        num_classes=num_classes
    )

    print(f"Model has {model.count_params():,} parameters")

    # Define callbacks
    checkpoint = ModelCheckpoint(
        '../models/skin_condition_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Reduced patience
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Reduced patience
        min_lr=1e-5,
        verbose=1
    )

    class MemoryCleanup(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            tf.keras.backend.clear_session()

    memory_cleanup = MemoryCleanup()

    # Create data generators that load from disk
    train_generator = DataChunkGenerator(
        os.path.join(os.path.dirname('../data/processed/train'), 'processed_chunks'),
        batch_size=batch_size,
        shuffle=True
    )

    val_generator = DataChunkGenerator(
        os.path.join(os.path.dirname('../data/processed/val'), 'processed_chunks'),
        batch_size=batch_size,
        shuffle=False
    )

    # Custom training loop for extreme memory optimization
    print("Step 3: Starting training with custom loop...")
    epochs = 10  # Reduced epochs for faster training

    train_steps = train_samples // batch_size
    val_steps = val_samples // batch_size

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        start_time = time.time()

        # Training phase
        train_loss = 0
        train_acc = 0
        train_batches = 0

        for x_batch, y_batch in train_generator:
            # Train on batch
            metrics = model.train_on_batch(x_batch, y_batch)
            train_loss += metrics[0]
            train_acc += metrics[1]
            train_batches += 1

            # Print progress
            if train_batches % 50 == 0:
                print(
                    f"  Batch {train_batches}/{train_steps} - loss: {train_loss / train_batches:.4f} - accuracy: {train_acc / train_batches:.4f}")

            if train_batches >= train_steps:
                break

        train_loss /= train_batches
        train_acc /= train_batches

        # Validation phase
        val_loss = 0
        val_acc = 0
        val_batches = 0

        for x_batch, y_batch in val_generator:
            # Validate on batch
            metrics = model.test_on_batch(x_batch, y_batch)
            val_loss += metrics[0]
            val_acc += metrics[1]
            val_batches += 1

            if val_batches >= val_steps:
                break

        val_loss /= val_batches
        val_acc /= val_batches

        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Print epoch summary
        time_taken = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs} - {time_taken:.1f}s - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print("Saving best model")
            model.save('../models/skin_condition_model_best.h5')
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Early stopping
                print("Early stopping triggered")
                break

        # Force garbage collection
        gc.collect()
        tf.keras.backend.clear_session()

    # Save final model
    model.save('../models/skin_condition_model_final.h5')

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('../models/training_history.png')
    plt.close()

    return model, class_labels


if __name__ == "__main__":
    print("Starting ultra-optimized training process...")
    train_model_efficient()