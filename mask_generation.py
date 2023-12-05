# Import libraries
from PIL import Image
import pandas as pd
import numpy as np
import os

# Define paths for the CSV file containing length-encoded masks
PATH_TO_CSV_DATA = r"train_ship_segmentations_prepared.csv"

# Define the destination folder where masks will be stored
DESTINATION_FOLDER = "airbus/train_v2_masks"

# Import the dataset containing image names and their corresponding length-encoded masks
df = pd.read_csv(PATH_TO_CSV_DATA)

def create_mask(mask_data, shape=(768, 768)):
    """
    Creates a mask based on the provided mask data, representing segmented image information.

    Args:
    - mask_data (str): A string representing mask data.
    - shape (tuple, optional): The shape of the mask to be created. Defaults to (768, 768).

    Returns:
    - img (numpy.ndarray): A mask represented as a NumPy array with values indicating pixel 
    presence (1) or absence (0) for the specified classes within the given shape.
    """
    if isinstance(mask_data, float):
        return np.zeros(shape, dtype=np.uint8)
        
    s = mask_data.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape).T

# Define a root folder for the created masks
os.makedirs(DESTINATION_FOLDER, exist_ok=True)

# Create a list with all prepared image names
images = df.groupby("ImageId").count().reset_index()["ImageId"].to_list()

# Iterate through the list of images, retrieve data from the CSV file, and create corresponding masks
for image in images:
    img_masks = df.loc[df["ImageId"] == image, "EncodedPixels"].to_list()
    all_masks = np.zeros((768, 768))
    for mask in img_masks:
        all_masks += create_mask(mask)
        
    mask_image = Image.fromarray(all_masks.astype(np.uint8) * 255)

    mask_name = image[:-4] + "_mask" + ".png"
    save_path = os.path.join(DESTINATION_FOLDER, mask_name)
    mask_image.save(save_path)
