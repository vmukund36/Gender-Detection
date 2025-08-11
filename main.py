#import the required libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from keras.preprocessing import image

#creating your training dataset
train_path_dir='your_training_folder_path'
test_path_dir='your_test_folder_path'
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(train_path_dir,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#creating your test dataset
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(test_path_dir,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#create your convolutional neural network based model 
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fit training data to train your model
cnn.fit(x = training_set, validation_data = test_set, epochs = 100)

#inference of model on the images of the dataset
image=[]
path= 'path to the dataset'
lenpath = len(path)
for img in os.listdir(path):
        img = os.path.join(path, img)
        test_image = tf.keras.preprocessing.image.load_img(img, target_size = (64, 64))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        image.append(test_image)
        result = cnn.predict(image)
        training_set.class_indices
        if result[0][0] == 0:
           prediction = 'woman'
        else:
           prediction = 'man'
        print(result)
        print(prediction)

#visualize metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(test_set, result)
print(cm)
accuracy_score(test_set, result)
