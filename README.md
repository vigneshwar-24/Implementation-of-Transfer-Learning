# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:


VGG19 is a variant of the VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layer, 5 MaxPool layers and 1 SoftMax layer).

Now we have use transfer learning with the help of VGG-19 architecture and use it to classify the CIFAR-10 Dataset

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Load CIFAR-10 Dataset & use Image Data Generator to increse the size of dataset

### STEP 3:
Import the VGG-19 as base model & add Dense layers to it

### STEP 4:
Compile and fit the model

### Step 5:
Predict for custom inputs using this model

## PROGRAM
Include your code here
```python

import pandas as pd
import numpy as np
from keras import layers
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras import utils
from sklearn.metrics import classification_report,confusion_matrix

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

image=x_train[1]
image.max()

train_generator=ImageDataGenerator(
    rotation_range=2,
    horizontal_flip=True,
    rescale=1.0/255.0,
    zoom_range=.1
)

test_generator=ImageDataGenerator(
    rotation_range=2,
    horizontal_flip=True,
    rescale=1.0/255.0,
    zoom_range=.1
)

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

base_model=VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3)
)

base_model.summary()

for layer in base_model.layers:
  layer.trainable = False
  
model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(2048,activation=('relu')))
model.add(Dense(1024,activation=('relu')))
model.add(Dense(512,activation=('relu')))
model.add(Dense(256,activation=('relu')))
model.add(Dense(128,activation=('relu')))
model.add(Dense(128,activation=('relu')))
model.add(Dense(64,activation=('relu')))
model.add(Dense(64,activation=('relu')))
model.add(Dense(32,activation=('relu')))
model.add(Dense(32,activation=('relu')))
model.add(Dense(10,activation=('softmax')))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics='accuracy'
)

batch_size=64
epochs=50

train_image_generator  = train_generator.flow(x_train,y_train_onehot,
                                         batch_size = batch_size)

test_image_generator  = test_generator.flow(x_test,y_test_onehot,
                                         batch_size = batch_size)
                                         
model.fit(train_image_generator,epochs=epochs,
          validation_data = test_image_generator)

metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(test_image_generator), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

```


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/vigneshwar-24/Implementation-of-Transfer-Learning/assets/77089276/a8813b1f-7c00-4b0d-8c15-506302bddb62)
![image](https://github.com/vigneshwar-24/Implementation-of-Transfer-Learning/assets/77089276/57f7be6b-184a-4667-a269-0f8b903d86ad)

### Classification Report
![image](https://github.com/vigneshwar-24/Implementation-of-Transfer-Learning/assets/77089276/9183f12f-789d-400a-acd3-738d88289c99)
### Confusion Matrix
![image](https://github.com/vigneshwar-24/Implementation-of-Transfer-Learning/assets/77089276/c94fcdf8-1a0e-4b4d-89af-63ab8adc386e)

## Conculsion
- We got an Accuracy of 60% with this model.
- There could be several reasons for not achieving higher accuracy.
- Here are a few possible explanations:
## Dataset compatibility:
- VGG19 was originally designed and trained on the ImageNet dataset, which consists of high-resolution images.
- In contrast, the CIFAR10 dataset contains low-resolution images (32x32 pixels).
- The difference in image sizes and content can affect the transferability of the learned features.
- Pretrained models like VGG19 might not be the most suitable choice for CIFAR10 due to this disparity in data characteristics.
## Inadequate training data:
- If the CIFAR10 dataset is relatively small, it may not provide enough diverse examples for the model to learn robust representations.
- Deep learning models, such as VGG19, typically require large amounts of data to generalize well.
- In such cases, you could consider exploring other architectures that are specifically designed for smaller datasets, or you might want to look into techniques - like data augmentation or transfer learning from models pretrained on similar datasets.
## Model capacity:
- VGG19 is a deep and computationally expensive model with a large number of parameters.
- If you are limited by computational resources or working with a smaller dataset, the model's capacity might be excessive for the task at hand.
- In such cases, using a smaller model architecture or exploring other lightweight architectures like MobileNet or SqueezeNet could be more suitable and provide better accuracy.
## RESULT
Thus, transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture is successfully implemented.
