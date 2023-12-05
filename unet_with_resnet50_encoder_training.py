# Import libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

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
INITIAL_LR = 1e-3
FINAL_LR = 1e-4
EPOCHS = 3
ALPHA = 0.5 # The balancing factor between positive and negative classes in focal loss function.
FOCAL_GAMMA = 2.0 # modulates the rate at which easy examples are down-weighted in focal loss function
BETA = 0.8 # Proportion coeficient for focal loss and log soft dice loss (1-pure focal loss ; 0-pure log soft dice loss)

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
    - x: A string representing a filename.
    - path: The root directory path to prepend (default is PATH_TO_IMAGES).

    Returns:
    - The combined filename and the provided relative path.
    """
    return path+x

def add_mask_root(x, path=PATH_TO_MASKS):
    """
    Args:
    - x: A string representing a filename.
    - path: The root directory path to prepend (default is PATH_TO_MASKS).

    Returns:
    - The combined filename and the provided relative path.
    """
    return path+x[:-4]+"_mask.png"

# Add columns with image relative file paths and mask relative file paths
X_train["image_filepaths"] = X_train["ImageId"].apply(add_root, path=PATH_TO_IMAGES)
X_train["mask_filepaths"] = X_train["ImageId"].apply(add_mask_root, path=PATH_TO_MASKS)
X_dev["image_filepaths"] = X_dev["ImageId"].apply(add_root, path=PATH_TO_IMAGES)
X_dev["mask_filepaths"] = X_dev["ImageId"].apply(add_mask_root, path=PATH_TO_MASKS)

# Consider only instances with ships
X_train = X_train[X_train["ship_present"] > 0]
X_dev = X_dev[X_dev["ship_present"] > 0]

print(f"\nNumber of instances in the train dataframe: {len(X_train)}")
print(f"Number of instances in the dev dataframe: {len(X_dev)}")

# Create a function for consistent augmentation of images and masks
def flip(img, mask):
    """
    Args:
    - img: An image tensor.
    - mask: A mask tensor associated with the image.

    Returns:
    - The image and mask tensors after applying a random flip operation (left-right and/or up-down).
    """
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    return img, mask

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

    img, mask = flip(img, mask)
    
    return img, mask

# Create TensorFlow datasets
X_train_segm = X_train[["image_filepaths", "mask_filepaths"]].values
X_train_segm_images, X_train_segm_masks = X_train_segm[:,0], X_train_segm[:,1]

X_dev_segm = X_dev[["image_filepaths", "mask_filepaths"]].values
X_dev_segm_images, X_dev_segm_masks = X_dev_segm[:,0], X_dev_segm[:,1]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_segm_images, X_train_segm_masks))
train_dataset = train_dataset.map(map_func=load_and_preprocess_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size=BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev_segm_images, X_dev_segm_masks))
dev_dataset = dev_dataset.map(map_func=load_and_preprocess_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
dev_dataset = dev_dataset.batch(batch_size=BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"\nCreate train dataset: {train_dataset.element_spec}")
print(f"Create dev dataset: {dev_dataset.element_spec}\n")

# Download and initialize pre-trained U-Net model with ResNet50 encoder
model = Unet(backbone_name='resnet50', encoder_weights='imagenet', encoder_freeze=True)

# Create focal loss function
class FocalLoss(tf.keras.losses.Loss):
    """
    Logarithmic Soft Dice Loss for evaluating segmentation tasks.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Calculate the Focal Loss between predicted and true segmentation masks.

        Args:
        - y_true: Ground truth segmentation mask.
        - y_pred: Predicted segmentation mask.

        Returns:
        - Focal Loss value.
        """
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        loss = -self.alpha * (1 - y_pred)**self.gamma * y_true * tf.math.log(y_pred + epsilon) \
               - (1 - self.alpha) * y_pred**self.gamma * (1 - y_true) * tf.math.log(1 - y_pred + epsilon)

        loss = tf.reduce_sum(loss, axis=-1)

        return loss

# Create log soft-dice loss function
class LogSoftDiceLoss(tf.keras.losses.Loss):
    """
    Logarithmic Soft Dice Loss for evaluating segmentation tasks.
    """
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        """
        Calculate the Logarithmic Soft Dice Loss between predicted and true segmentation masks.

        Args:
        - y_true: Ground truth segmentation mask.
        - y_pred: Predicted segmentation mask.

        Returns:
        - Logarithmic Soft Dice Loss value.
        """
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)

        intersection = tf.reduce_sum(y_true * y_pred)
        sum_prediction = tf.reduce_sum(y_pred)
        sum_true = tf.reduce_sum(y_true)

        dice_coefficient = (2.0 * intersection + 1e-8) / (sum_true + sum_prediction + 1e-8)

        log_soft_dice_loss = -tf.math.log(dice_coefficient + 1e-8)

        return log_soft_dice_loss

# Create combined focal and log soft-dice loss function
class CombinedLoss(tf.keras.losses.Loss):
    """
    Combination of Focal Loss and Logarithmic Soft Dice Loss.
    """
    def __init__(self, alpha=0.5, beta=0.5, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_gamma = focal_gamma
        self.focal_loss = FocalLoss(alpha=alpha, gamma=focal_gamma)
        self.log_soft_dice_loss = LogSoftDiceLoss()

    def call(self, y_true, y_pred):
        """
        Calculate the Combined Loss, a weighted sum of Focal Loss and Logarithmic Soft Dice Loss.

        Args:
        - y_true: Ground truth segmentation mask.
        - y_pred: Predicted segmentation mask.

        Returns:
        - Combined Loss value.
        """
        focal_loss_value = self.focal_loss(y_true, y_pred)
        log_soft_dice_loss_value = self.log_soft_dice_loss(y_true, y_pred)

        combined_loss = self.beta * focal_loss_value + (1 - self.beta) * log_soft_dice_loss_value

        return combined_loss

# Define dice score metric
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

# Create a model checkpoint only for model weights
checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                             monitor='val_dice_score', 
                             save_best_only=True, 
                             save_weights_only=True, 
                             mode='max',  
                             verbose=0)

# Create a learning rate scheduler for reducing the learning rate during training
lr_scheduler = LearningRateScheduler(lambda epoch: INITIAL_LR  + (FINAL_LR - INITIAL_LR) * epoch / EPOCHS)

# Create an instance of the dice score metric
dice_score = DiceScore()

# Create an instance of the combined loss function
combined_focal_dice_loss = CombinedLoss(alpha=ALPHA, beta=BETA, focal_gamma=FOCAL_GAMMA)

# Compile the model
model.compile(loss=combined_focal_dice_loss,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[dice_score])

# Train U-Net model
history_unet = model.fit(train_dataset,
                         epochs=EPOCHS,
                         steps_per_epoch=int(len(train_dataset)),
                         validation_data=dev_dataset,
                         validation_steps=int(len(dev_dataset)),
                         callbacks=[lr_scheduler, checkpoint])
