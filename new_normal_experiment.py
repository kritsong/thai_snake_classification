import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical  # ✅ ADDED for one-hot encoding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import json
import shutil  # ❌ No longer needed for data loading, but kept for prudence

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} Physical GPUs configured.")
    except RuntimeError as e:
        print(f"RuntimeError during GPU configuration: {e}")
else:
    print("No GPUs found. Running on CPU.")

# --- Reproducibility ---
tf.random.set_seed(42)
np.random.seed(42)

# --- Configuration Parameters ---
data_dir = "C:\\Users\\ADMIN\\Downloads\\snake_datasetNew"
# ✅ RENAMED output dir to reflect the change
base_output_dir = os.path.join(os.getcwd(), 'cnn_experiment_results_ram_based')
thresholds = [100, 200, 300, 400, 500]

base_augmentation = {
    'rotation_range': 10, 'width_shift_range': 0.1, 'height_shift_range': 0.1,
    'zoom_range': 0.1, 'horizontal_flip': True, 'shear_range': 0.1, 'fill_mode': 'nearest'
}

augmentation_levels = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
EPOCHS = 50
BATCH_SIZE = 32
DEFAULT_INPUT_SIZE = (224, 224)


# --- Custom Callback for Bulletproof History Logging (Unchanged) ---
class CustomHistoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(CustomHistoryLogger, self).__init__()
        self.filepath = filepath
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if hasattr(value, 'numpy'):
                value = value.numpy()
            if isinstance(value, np.ndarray):
                value = np.mean(value)
            self.history.setdefault(key, []).append(float(value))

    def on_train_end(self, logs=None):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.history, f, indent=4)
            print(f"Custom history successfully saved to {self.filepath}")
        except Exception as e:
            print(f"FATAL: Custom history logger failed to save file. Error: {e}")


def create_augmented_settings(base_params, scale):
    return {k: v * scale if isinstance(v, (int, float)) else v for k, v in base_params.items()}


augmentation_settings = {
    'none': {},
    'low': create_augmented_settings(base_augmentation, augmentation_levels['low']),
    'medium': create_augmented_settings(base_augmentation, augmentation_levels['medium']),
    'high': create_augmented_settings(base_augmentation, augmentation_levels['high']),
}


# --- ✅ NEW: Data Loading Function (Loads Images into RAM) ---
def load_and_filter_dataset_ram(data_dir_path, threshold_value, input_size):
    """
    Loads images directly into a NumPy array in memory instead of just paths.
    """
    print(f"\n[Loading Dataset into RAM] Threshold: {threshold_value}")
    all_images_list, all_labels_int = [], []
    class_map = {}

    # First, find which classes meet the threshold
    valid_class_names = []
    for class_folder_name in sorted(os.listdir(data_dir_path)):
        class_folder_path = os.path.join(data_dir_path, class_folder_name)
        if os.path.isdir(class_folder_path):
            image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(image_files) >= threshold_value:
                valid_class_names.append(class_folder_name)

    if not valid_class_names:
        print("No classes met the threshold. Returning empty dataset.")
        return np.array([]), np.array([]), []

    class_to_idx = {name: i for i, name in enumerate(valid_class_names)}

    # Now, load images only from valid classes
    for class_name in valid_class_names:
        class_idx = class_to_idx[class_name]
        class_folder_path = os.path.join(data_dir_path, class_name)
        image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for file_name in image_files:
            try:
                img_path = os.path.join(class_folder_path, file_name)
                # Load, convert, resize, and append image data
                img = Image.open(img_path).convert('RGB').resize(input_size)
                all_images_list.append(np.array(img))
                all_labels_int.append(class_idx)
            except Exception as e:
                print(f"    ! Skipping image '{file_name}': {e}")

    if not all_images_list:
        return np.array([]), np.array([]), []

    # Convert lists to NumPy arrays
    images_array = np.array(all_images_list, dtype=np.uint8)
    # Convert integer labels to one-hot encoded format for Keras
    labels_one_hot = to_categorical(all_labels_int, num_classes=len(valid_class_names))

    print(f"  Loaded {len(images_array)} samples into RAM across {len(valid_class_names)} classes.")
    return images_array, labels_one_hot, valid_class_names


# --- ✅ MODIFIED: Main Training Loop (to use RAM data) ---
def run_cnn_training_only():
    os.makedirs(base_output_dir, exist_ok=True)

    models_config = {
        'MobileNetV3Small': (MobileNetV3Small, DEFAULT_INPUT_SIZE, mobilenet_preprocess),
        'ResNet50': (ResNet50, DEFAULT_INPUT_SIZE, resnet_preprocess),
        'EfficientNetB0': (EfficientNetB0, DEFAULT_INPUT_SIZE, efficientnet_preprocess),
    }

    for current_threshold in thresholds:
        # Load all data for the current threshold into memory
        X_raw, y_one_hot, class_names = load_and_filter_dataset_ram(data_dir, current_threshold, DEFAULT_INPUT_SIZE)

        if X_raw.size == 0 or len(class_names) < 2:
            print(f"Skipping threshold {current_threshold} due to insufficient data.")
            continue

        num_classes = len(class_names)

        # Split the in-memory arrays
        X_train_raw, X_val_raw, y_train_one_hot, y_val_one_hot = train_test_split(
            X_raw, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
        )

        # Calculate class weights from the training split
        y_train_int = np.argmax(y_train_one_hot, axis=1)
        class_weights_dict = {i: w for i, w in enumerate(compute_class_weight(
            'balanced', classes=np.unique(y_train_int), y=y_train_int
        ))}

        # ❌ REMOVED: No longer need to create temporary directories and copy files
        # temp_data_dir, train_dir, val_dir are all gone.

        for model_name, (base_model_class, input_size, preprocess_fn) in models_config.items():

            # Preprocess the validation data once, as it doesn't need augmentation
            X_val_processed = preprocess_fn(X_val_raw.copy())

            for aug_name, aug_params in augmentation_settings.items():
                for train_from_scratch in [True, False]:
                    fine_tune_options = [None] if train_from_scratch else [False, True]
                    for fine_tune in fine_tune_options:
                        if train_from_scratch:
                            status_str, lr = "full_train", 1e-3
                        else:
                            status_str, lr = ("frozen", 5e-5) if not fine_tune else ("fineT", 2e-5)

                        exp_name = f"{model_name}_aug{aug_name}_{status_str}_thres{current_threshold}_weightsTrue"
                        exp_path = os.path.join(base_output_dir, exp_name)

                        h5_path = os.path.join(exp_path, 'last_model.h5')
                        hist_path = os.path.join(exp_path, 'history.json')

                        if os.path.exists(h5_path) or os.path.exists(hist_path):
                            print(f"\nSkipping existing experiment: {exp_name}")
                            continue

                        print(f"\n--- Running experiment: {exp_name} ---")
                        os.makedirs(exp_path, exist_ok=True)

                        base_model = base_model_class(
                            weights='imagenet' if not train_from_scratch else None,
                            include_top=False, input_shape=(*input_size, 3)
                        )
                        if not train_from_scratch: base_model.trainable = fine_tune

                        x = GlobalAveragePooling2D()(base_model.output)
                        x = Dense(1024, activation='relu')(x)
                        predictions = Dense(num_classes, activation='softmax')(x)
                        model = Model(inputs=base_model.input, outputs=predictions)

                        model.compile(
                            optimizer=Adam(learning_rate=ExponentialDecay(lr, 10000, 0.96, True)),
                            loss='categorical_crossentropy', metrics=['accuracy']
                        )

                        callbacks = [
                            tf.keras.callbacks.ModelCheckpoint(filepath=h5_path, save_weights_only=True),
                            CustomHistoryLogger(filepath=hist_path)
                        ]

                        # Create a generator that augments and preprocesses data from RAM
                        train_datagen = ImageDataGenerator(
                            preprocessing_function=preprocess_fn,
                            **aug_params
                        )

                        # Use .flow() for in-memory data, not .flow_from_directory()
                        train_generator = train_datagen.flow(
                            X_train_raw,
                            y_train_one_hot,
                            batch_size=BATCH_SIZE
                        )

                        model.fit(
                            train_generator,
                            steps_per_epoch=max(1, len(X_train_raw) // BATCH_SIZE),
                            # Provide the preprocessed validation set directly
                            validation_data=(X_val_processed, y_val_one_hot),
                            epochs=EPOCHS,
                            class_weight=class_weights_dict,
                            callbacks=callbacks,
                            verbose=1
                        )

        # ❌ REMOVED: No temporary directory to clean up


if __name__ == "__main__":
    run_cnn_training_only()