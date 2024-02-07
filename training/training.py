import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers, models


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 5

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./PlantVillage",
    shuffle= True,
    image_size= (IMAGE_SIZE,IMAGE_SIZE),
    batch_size= BATCH_SIZE 
)

class_names = dataset.class_names

print(class_names)
n_classes = len(class_names)
print(len(dataset))

#Datavisualisation 

plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(image_batch[0].numpy())
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    #plt.show()
    print(label_batch.numpy())
    
    
#Train test split

"""
   80% -> training
   20% -> 10% validation 10% testing 
"""
import math
train_size = 0.8
#Train dataset
train_ds = dataset.take(math.floor(len(dataset)*train_size))
trainds_size = len(train_ds)

testval_ds = dataset.skip(trainds_size)

val_size = 0.1
valds_size = math.floor(val_size*len(dataset))

#Validation dataset
val_ds =testval_ds.take(valds_size) 

#Test dataset
test_ds = testval_ds.skip(valds_size)


def get_dataset_partitions_tf(ds, train_split = 0.8, val_split = 0.1, test_split = 0.1, shuffle = True, shuffle_size = 10000):
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)
        
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds
    
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

#Input pipeline for efficiency ---> caching

train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

# Preprocessing layers

resize_and_rescale = tf.keras.Sequential([
    
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

# Data augmentation layers

data_augmentation = tf.keras.Sequential([
    
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])


# Training Process

input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE, CHANNELS)
# Convolutional NN
cnn_model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
      
])

cnn_model.build(input_shape=input_shape)

cnn_model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = cnn_model.fit(
    train_ds, 
    epochs = EPOCHS, 
    batch_size=BATCH_SIZE,
    verbose = 1,
    validation_data=val_ds
)

params = history.history

acc = params['accuracy']
val_acc = params['val_accuracy']
loss = params['loss']
val_loss = params['val_loss']


#Visualizing accuracy and loss for training and validation
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss, label = 'Validation Loss')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Loss')

plt.show()
plt.close()
import numpy as np
for image_batch, labels_batch in test_ds.take(1):
    first_image = image_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0]
    
    print("first image to predict")
    plt.imshow(first_image)
    print("Actual label: ", class_names[first_label.numpy()])
    
    batch_prediction = cnn_model.predict(image_batch)
    
    print("Predicted label: ", class_names[np.argmax(batch_prediction[0])])
    
    
def predict(model, images):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    return predicted_class, confidence
plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(cnn_model, images)
        
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\n Predicted class: {predicted_class}.\n Confidence : {confidence}")
        
        
        plt.axis("off")
        
plt.show()

        
import os        
dirs = [int(i) for i in os.listdir("../models") + [0]]
model_version = max(dirs)+1
cnn_model.save(f"../models/{model_version}")
    