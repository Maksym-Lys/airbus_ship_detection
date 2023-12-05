# Import libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from segmentation_models import Unet
from segmentation_models.utils import set_trainable

# Set up GPU memory configuration
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# Define global variables and hyperparameters
CHECKPOINT_PATH = r"models/unet_resnet50/unet_resnet50.ckpt"
PATH_TO_CSV_DATA = r"train_ship_segmentations_prepared.csv"
PATH_TO_IMAGES = r"airbus/train_v2/"
PATH_TO_MASKS = r"airbus/train_v2_masks/"
BATCH_SIZE = 16
IMG_SHAPE = 256

# Read prepared metadata
df = pd.read_csv(PATH_TO_CSV_DATA)

# Group by unique image names and preserve other columns' information
prepared_df = df.groupby('ImageId')[["ship_count", "ship_present"]].max().reset_index()

# Split images and labels into training, development, and testing sets
X_train, X_val = train_test_split(prepared_df, test_size=0.1, random_state=42, stratify=prepared_df["ship_count"])
X_dev, X_test = train_test_split(X_val, test_size=0.5, random_state=42, stratify=X_val["ship_count"])

def add_root(x, path=PATH_TO_IMAGES):
    """
    Args:
    - x: A string representing a relative path.
    - path: The root directory path to prepend (default is PATH_TO_IMAGES).

    Returns:
    - The combined filename and the provided relative path.
    """
    return path+x

def add_mask_root(x, path=PATH_TO_IMAGES):
    """
    Args:
    - x: A string representing a filename.
    - path: The root directory path to prepend (default is PATH_TO_MASKS).

    Returns:
    - The combined filename and the provided relative path.
    """
    return path+x[:-4]+"_mask.png"

# Add columns with image relative file paths and mask relative file paths
X_test["image_filepaths"] = X_test["ImageId"].apply(add_root, path=PATH_TO_IMAGES)
X_test["mask_filepaths"] = X_test["ImageId"].apply(add_mask_root, path=PATH_TO_MASKS)

# Consider only instances with ships
X_test = X_test[X_test["ship_present"] > 0]

print(f"\nNumber of instances in the test dataframe: {len(X_test)}")

# Create a data preprocessing function
def load_and_preprocess_image_and_mask(image_path, mask_path):  
    """
    Args:
    - image_path: Path to the image file.
    - mask_path: Path to the mask file associated with the image.

    Returns:
    - Preprocessed image and mask tensors.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SHAPE, IMG_SHAPE])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SHAPE, IMG_SHAPE], method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.
    
    return img, mask

# Create TensorFlow dataset
X_test_segm = X_test[["image_filepaths", "mask_filepaths"]].values
X_test_segm_images, X_test_segm_masks = X_test_segm[:,0], X_test_segm[:,1]

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_segm_images, X_test_segm_masks))
test_dataset = test_dataset.map(map_func=load_and_preprocess_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size=BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"\nCreate test dataset: {test_dataset.element_spec}\n")

class DiceScore(tf.keras.metrics.Metric):
    """
    Custom metric to compute the Dice coefficient for evaluating segmentation tasks.
    """
    def __init__(self, name='dice_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.sum_true = self.add_weight(name='sum_true', initializer='zeros')
        self.sum_pred = self.add_weight(name='sum_pred', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the Dice coefficient metric.

        Args:
        - y_true: Ground truth segmentation mask.
        - y_pred: Predicted segmentation mask.
        - sample_weight: Optional sample weights.

        Updates:
        - Internal states (intersection, sum_true, sum_pred) of the metric.
        """
        y_pred = tf.cast(tf.math.round(y_pred), tf.float32) 
        y_true = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
        y_pred = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        sum_true = tf.reduce_sum(y_true)
        sum_pred = tf.reduce_sum(y_pred)

        self.intersection.assign_add(intersection)
        self.sum_true.assign_add(sum_true)
        self.sum_pred.assign_add(sum_pred)

    def result(self):
        """
        Compute the Dice coefficient.

        Returns:
        - Dice coefficient value.
        """
        dice_coefficient = (2.0 * self.intersection + 1e-8) / (self.sum_true + self.sum_pred + 1e-8)
        return dice_coefficient

    def reset_state(self):
        """
        Reset the state of the metric.
        """
        self.intersection.assign(0.0)
        self.sum_true.assign(0.0)
        self.sum_pred.assign(0.0)

# Download and initialize pre-trained model
model = Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)

# Create an instance of the dice score metric
dice_score = DiceScore()

# Compile model
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=[dice_score])

# Load the model weights
model.load_weights(CHECKPOINT_PATH)

# Calculate evaluation results
results = model.evaluate(test_dataset)
print(f"""\nEvaluation Results:
Dice score: {results[1]:.4f}
""")
