# Airbus Ship Detection

- Link on Kaggel: https://www.kaggle.com/c/airbus-ship-detection/overview
- Link to task zip data in Google Drive (prepared CSV file, model weights and masks): https://drive.google.com/file/d/1poiwUNB0C4qmgs-oLlCYzh1pRzntQhXF/view?usp=sharing

### Exploratory data analysis:

   In this notebook, a brief analysis of the Airbus Ship Detection dataset was conducted. 
It was found that only 22% of the images in the dataset contain ships, with varying numbers 
(from 1 to 15) and different forms and sizes. The ratio of 'ship' class pixels to 'background' 
class pixels across the entire dataset is 0.001128. However, when considering only images 
with ships, this ratio improves to 0.005103, which is still quite imbalanced.
   Based on this information, it seems appropriate to create two models: one for classification 
(to determine if a ship is present in an image) and another for segmentation. Training the 
segmentation model solely on images containing ships could significantly enhance its performance.
   Additionally, it was observed that some images are corrupted, exhibiting regions with solid 
blue or black pixels. One image, '6384c3e78.jpg,' is truncated. A function was written to 
identify these corrupted images and add them to the list of flagged images.
   A new dataset was then generated, excluding the corrupted images, and a label column indicating 
the presence (1) or absence (0) of a ship was added. This revised dataset was saved as a CSV file.

### Mask generation:

   This Python script generates a folder containing PNG masks for respective images using length-encoded 
masks from a CSV file. Pre-generating these masks in advance can be advantageous as it eliminates 
the necessity of writing complex preprocessing TensorFlow functions. This approach can enhance model 
training speed and is particularly useful when conducting multiple experiments with various hyperparameters.

### ResNet50 classification model training:

   For training ResNet50 classification model training and developing datasets should be created.
The dataset was split into train, dev, and test parts using a defined random state and stratification based 
on the number of ships, with proportions of 0.9, 0.05, and 0.05 respectively. The test set will solely be 
used for inference purposes.

   TensorFlow datasets were prepared with prefetched data having image shapes of 256x256x3 and a batch size of 32.

   Initially, the base ResNet50 model was frozen, and a head for binary classification task was trained for 
3 epochs. This training utilized the Adam optimizer with a learning rate of 1e-3. The loss function used was 
binary cross-entropy, suitable for the sigmoid activation function. At the end of each epoch, a model 
checkpoint callback was implemented to save the best weights based on validation loss.

   Following these 3 epochs, all layers in the last block of the base ResNet50 were unfrozen, except for 
the batch normalization layers to ensure convergence stability. The model was then fine-tuned for an 
additional 3 epochs with a reduced learning rate of 1e-4.

### ResNet50 classification model inference:

The ResNet50 classification model used the same data split as the training script. The dataset was divided into 
train, dev, and test parts using a defined random state and stratification based on the number of ships, preserving 
proportions of 0.9, 0.05, and 0.05, respectively. Only the test part was utilized for inference.

Inference Results:
Accuracy: 0.9512
Precision: 0.9431
Recall: 0.8287
F1-score: 0.884


### U-Net segmentation model with ResNet50 encoder training (only on images with ships):

   For the U-Net model, the same data split as for the classification model was utilized. The dataset was divided into 
train, dev, and test parts, employing a defined random state and stratification based on the number of ships, 
maintaining proportions of 0.9, 0.05, and 0.05, respectively. Subsequently, all samples without ships were removed.
   Image filepaths and mask filepaths were integrated into the datasets for the preprocessing function. This function received 
these filepaths, loaded images and corresponding masks, resized and rescaled them, and applied augmentation techniques.
   TensorFlow datasets were structured with prefetched data in tuples, featuring image shapes of 256x256x3 and mask shapes 
of 256x256x1, with a batch size of 16.
   For the segmentation model, U-Net with a ResNet50 encoder was chosen. The loss function comprised a combination of focal 
loss and logarithmic soft dice loss functions, controlled by a proportion coefficient beta  (1-pure focal loss ; 0-pure 
logarithmic soft dice loss), which balanced between the two functions. The focal loss function involved parameters alpha 
and gamma, with alpha adjusting the balance between positive and negative classes, while gamma modulated the rate at which 
easy examples were down-weighted. This combination of losses aimed to address the challenge of highly unbalanced classes, 
where the focal loss function excels, while the logarithmic soft dice loss function demonstrates good results for dice score, 
but potentially convergently unstable when used alone.
   During inference, the metric used was the dice score. The model underwent training for 3 epochs, employing the Adam optimizer 
with a learning rate decay from 1e-3 to 1e-4. Additionally, a checkpoint callback was implemented to save the best weights 
based on validation loss.

### U-Net segmentation model with ResNet50 encoder inference (only on images with ships):

   The U-Net segmentation model used the same data split as the training script. The dataset was split into train, dev, and test parts 
using a defined random state and stratification based on the number of ships, preserving proportions of 0.9, 0.05, and 0.05, respectively. 
Only samples from test data, containing images and masks with ships, was employed for inference.

Inference Results:
Dice score: 0.8395

### Combined ResNet50 classification model and U-Net segmentation model inference on all dataset:

   The Combined ResNet50 classification model and U-Net segmentation model followed the same data split as in the training script. 
The dataset was divided into train, dev, and test parts, using a defined random state and stratification based on the number of 
ships, maintaining proportions of 0.9, 0.05, and 0.05, respectively.
   Both the classification and segmentation models were used sequentially to process the entire dataset. Initially, the 
classification model determined if ships were present in the image. If no ships were detected, the algorithm returned an 
empty 256x256x1 mask. However, if a ship was found, the image proceeded to the segmentation model to predict its mask. 
The masks predicted by this two-staged model were then compared with the true mask, and the dice score was calculated.

Inference Results:
Dice score: 0.8192

### Combined ResNet50 classification model and U-Net segmentation model demonstration:

   In the model demo, there's a cell where you can select a random image and specify the number of images for mask prediction. 
It fetches a random image from the test dataset and predicts the corresponding mask. After prediction, pairs of images and 
images with masks overlaid on them are displayed.