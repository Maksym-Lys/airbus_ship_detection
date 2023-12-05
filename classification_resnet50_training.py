# Import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint

# Set up mixed precision for calculation speedup and add a random seed for repeatability
mixed_precision.set_global_policy(policy="mixed_float16")
tf.random.set_seed(42)

# Set up GPU memory configuration
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# Define global variables and hyperparameters
CHECKPOINT_PATH = r"models/resnet50classifier/resnet50classifier.ckpt"
PATH_TO_IMAGES = r"airbus/train_v2/"
PATH_TO_CSV_DATA = r"train_ship_segmentations_prepared.csv"
BATCH_SIZE = 32
IMG_SHAPE = 256
EPOCHS = 3
FINE_TUNE_EPOCHS = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4

# Read prepared metadata
df = pd.read_csv(PATH_TO_CSV_DATA)
prepared_df = df.groupby('ImageId')[["ship_count", "ship_present"]].max().reset_index()

def add_root(x, path=PATH_TO_IMAGES):
    """
    Args:
    - x: A string representing a filename.
    - path: The root directory path to prepend (default is PATH_TO_IMAGES).

    Returns:
    - The combined filename and the provided relative path.
    """
    return path+x

# Add a column with image relative file paths
prepared_df["filepaths"] = prepared_df["ImageId"].apply(add_root)

# Split images and labels into training, development, and testing sets
X_train, X_val = train_test_split(prepared_df, test_size=0.1, random_state=42, stratify=prepared_df["ship_count"])
X_dev, X_test = train_test_split(X_val, test_size=0.5, random_state=42, stratify=X_val["ship_count"])

print(f"\nNumber of instances in the train dataframe: {len(X_train)}")
print(f"Number of instances in the dev dataframe: {len(X_dev)}")

# Create a data preprocessing function
@tf.function
def load_and_preprocess_image(image_path, label):
    """
    Reads and preprocesses an image from the provided file path.

    Args:
    - image_path: A string representing the file path of the image.
    - label: The label associated with the image.

    Returns:
    - img: The preprocessed image as a TensorFlow Tensor.
    - label: The associated label.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SHAPE, IMG_SHAPE])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, label

# Create TensorFlow datasets
train_image_filepaths = X_train["filepaths"].values
train_labels = X_train["ship_present"].values

dev_image_filepaths = X_dev["filepaths"].values
dev_labels = X_dev["ship_present"].values

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_filepaths, train_labels))
train_dataset = train_dataset.map(map_func=load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size=BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

dev_dataset = tf.data.Dataset.from_tensor_slices((dev_image_filepaths, dev_labels))
dev_dataset = dev_dataset.map(map_func=load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dev_dataset = dev_dataset.batch(batch_size=BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"\nCreate train dataset: {train_dataset.element_spec}")
print(f"Create dev dataset: {dev_dataset.element_spec}\n")

# Define input image shape
input_shape = (IMG_SHAPE, IMG_SHAPE, 3)

# Download and initialize pre-trained model
base_model = tf.keras.applications.resnet.ResNet50(include_top=False)
base_model.trainable = False

# Create a ResNet50 model with a head for 2 classes
inputs = layers.Input(shape=input_shape, name="input_layer")
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dense(1, activation='sigmoid', dtype=tf.float32)(x)
model = tf.keras.Model(inputs, x)

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=["accuracy"])

# Create a model checkpoint only for model weights
checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                             monitor='val_loss', 
                             save_best_only=True, 
                             save_weights_only=True, 
                             mode='min',  
                             verbose=0)

# Train the model
history_resnet50 = model.fit(train_dataset, 
                             epochs=EPOCHS,
                             steps_per_epoch=int(len(train_dataset)),
                             validation_data=dev_dataset,
                             validation_steps=int(len(dev_dataset)),
                             callbacks=[checkpoint])

# Unfreeze all layers in 'conv-5' except batch normalization layers
base_model.trainable = False
for layer in base_model.layers:
    if "conv5" in layer.name:
        if "bn" not in layer.name:
            layer.trainable = True

# Recompile the model for fine-tuning
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
              metrics=["accuracy"])

print("fine-tuning stage stage")
# Fine-tune the model
history_fine_tune_resnet50 = model.fit(train_dataset,
                                       epochs=FINE_TUNE_EPOCHS + EPOCHS - 1,
                                       steps_per_epoch=int(len(train_dataset)),
                                       validation_data=dev_dataset,
                                       validation_steps=int(len(dev_dataset)),
                                       initial_epoch=history_resnet50.epoch[-1],
                                       callbacks=[checkpoint])
