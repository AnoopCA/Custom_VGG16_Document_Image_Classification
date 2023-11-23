import warnings
warnings.filterwarnings("ignore")
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU for training

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class VGG16Model:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=(1000, 754, 3), filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=512, activation='relu')) #4096
        model.add(Dense(units=512, activation='relu')) #4096
        model.add(Dense(units=9, activation='softmax'))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, train_dir, test_dir, epochs=1, steps_per_epoch=1824, validation_steps=10):
        trdata = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        traindata = trdata.flow_from_directory(directory=train_dir, target_size=(1000, 754), batch_size=1)
        tsdata = ImageDataGenerator(rescale=1./255)
        testdata = tsdata.flow_from_directory(directory=test_dir, target_size=(1000, 754), batch_size=1)

        hist = self.model.fit_generator(steps_per_epoch=steps_per_epoch, generator=traindata, validation_data=testdata, 
                                        validation_steps=validation_steps, epochs=epochs)

        return hist

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)

if __name__ == '__main__':
    vgg16_model = VGG16Model()
    train_dir = r"D:\ML_Projects\Custom_VGG16_Document_Image_Classification\train"
    test_dir = r"D:\ML_Projects\Custom_VGG16_Document_Image_Classification\test"
    history = vgg16_model.train(train_dir=train_dir, test_dir=test_dir)
    vgg16_model.save_model("vgg16_model_1.h5")

    print("accuracy:", np.mean(history.history["accuracy"]))
    print("val_accuracy:", np.mean(history.history['val_accuracy']))
    print("loss:", np.mean(history.history['loss']))
    print("val_loss:", np.mean(history.history['val_loss']))

    # Load the model and use it for predictions
    # vgg16_model.load_model("vgg16_model_1.h5")
    # img = image.load_img("test/2074221556.tif", target_size=(224, 224))
    # img = np.asarray(img)
    # img = np.expand_dims(img, axis=0)
    # output = vgg16_model.model.predict(img)
    # print(output)
