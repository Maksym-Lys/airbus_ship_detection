# Import libraries
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up GPU memory configuration
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# Define global variables and hyperparameters
CHECKPOINT_PATH = r"models/resnet50classifier/resnet50classifier.ckpt"
PATH_TO_IMAGES = r"airbus/train_v2/"
PATH_TO_CSV_DATA = r"train_ship_segmentations_prepared.csv"
BATCH_SIZE = 32
IMG_SHAPE = 256

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

print(f"\nNumber of instances in the test dataframe: {len(X_test)}")

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
test_image_filepaths = X_test["filepaths"].values
test_labels = X_test["ship_present"].values

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_filepaths, test_labels))
test_dataset = test_dataset.map(map_func=load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size=BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"\nCreate test dataset: {test_dataset.element_spec}\n")

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

# Create additional metrics for inference
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + tf.keras.backend.epsilon()))

# Compile model
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=["accuracy", precision, recall, f1_score])

# Load the model weights
model.load_weights(CHECKPOINT_PATH)

# Calculate evaluation results
results = model.evaluate(test_dataset)
print(f"""\nEvaluation Results:
Accuracy: {results[1]:.4f}
Precision: {results[2]:.4f}
Recall: {results[3]:.4f}
F1 Score: {results[4]:.4f}
""")
