# Version:  2023 Dinh Phung, 2022 Trung Le

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

RANDOM_SEED = 3181

class DatasetManager():
    def __init__(self, dataset_name, data_dir):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.image_size = (32, 32)

    def preprocess_image(self, data):
        image = data['image']
        label = data['label']
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
        
    def preprocess_dataset(self):
        self.ds_train = self.ds_train.map(self.preprocess_image)
        self.ds_val = self.ds_val.map(self.preprocess_image)
        self.ds_test = self.ds_test.map(self.preprocess_image)

    def load_dataset(self):
        # Split the dataset into training and validation
        train_split = "train[:90%]"
        validation_split = "train[90%:]"
        test_split = "test"

        self.ds_train, self.ds_info = tfds.load(self.dataset_name, data_dir=self.data_dir, split=train_split, shuffle_files=True, with_info=True)
        self.ds_val = tfds.load(self.dataset_name, data_dir=self.data_dir, split=validation_split, shuffle_files=True)
        self.ds_test = tfds.load(self.dataset_name, data_dir=self.data_dir, split=test_split)

        self.n_classes = self.ds_info.features['label'].num_classes
    
    def show_examples(self):
        tfds.show_examples(self.ds_train, self.ds_info)


class BaseImageClassifier():
    def __init__(self, 
                 name='Base Classifier',
                 width=32, height=32, depth=3,
                 num_blocks=2,
                 feature_maps=32,
                 num_classes=4, 
                 drop_rate=0.2,
                 batch_norm=None,
                 is_augmentation=False,
                 activation_func='relu',
                 use_skip=True,
                 optimizer='adam',
                 batch_size=32,
                 num_epochs=20,
                 learning_rate=0.0001,
                 verbose= True):

        assert (0<<num_blocks<=min(width, height))
        self.name=name
        self.width=width
        self.height=height
        self.depth=depth
        self.num_blocks=num_blocks

        # number of feature maps will double for each incremental blocks
        self.feature_maps=[feature_maps*(1 << i) for i in range(num_blocks)] 

        self.num_classes=num_classes
        self.drop_rate=drop_rate
        self.batch_norm=batch_norm
        self.use_skip=use_skip
        self.is_augmentation=is_augmentation
        self.activation_func=activation_func
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.verbose=verbose
        if optimizer=='adam':
            self.optimizer = keras.optimizers.Adam(learning_rate)
        elif optimizer=='nadam':
            self.optimizer=keras.optimizers.Nadam(learning_rate)
        elif optimizer=='adagrad':
            self.optimizer=keras.optimizers.Adagrad(learning_rate)
        elif optimizer=='rmsprop':
            self.optimizer=keras.optimizers.RMSprop(learning_rate)
        elif optimizer=='adadelta':
            self.optimizer=keras.optimizers.Adadelta(learning_rate)
        else:
            self.optimizer=keras.optimizers.SGD(learning_rate, momentum=0.9)

        self.model = models.Sequential()
        self.history = None
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
    
    def build_cnn(self):
       self.model = models.Sequential()
       self.model.add(layers.Conv2D(32, (3,3), padding='same', activation=self.activation_func, input_shape=(self.height,self.width,self.depth)))
       self.model.add(layers.Conv2D(32, (3,3), padding='same', activation=self.activation_func))
       self.model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
       self.model.add(layers.Conv2D(64, (3,3), padding='same', activation=self.activation_func))
       self.model.add(layers.Conv2D(64, (3,3), padding='same', activation=self.activation_func))
       self.model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
       self.model.add(layers.Flatten())
       self.model.add(layers.Dense(self.num_classes, activation='softmax'))
       self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def summary(self):
        print(self.model.summary())
    
    def optimize_data_pipeline(self, dataset, batch_size=32):
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def fit(self, ds_train, ds_val, num_epochs=None):
        num_epochs = self.num_epochs if num_epochs is None else num_epochs
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(ds_train, epochs=num_epochs, validation_data=ds_val, verbose=self.verbose)
    
    def compute_accuracy(self, ds_test):
        evaluation_results = self.model.evaluate(ds_test)
        metrics = ['loss', 'accuracy']
        # Print the evaluation results
        for metric, result in zip(metrics, evaluation_results):
            print(f'{metric}: {result}')
    
    def plot_progress(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.history.history['accuracy'], label='train accuracy', color='green', marker="o")
        ax1.plot(self.history.history['val_accuracy'], label='valid accuracy', color='blue', marker = "v")
        ax2.plot(self.history.history['loss'], label = 'train loss', color='orange', marker="o")
        ax2.plot(self.history.history['val_loss'], label = 'valid loss', color='red', marker = "v")
        ax1.legend(loc=3)

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color='g')
        ax2.set_ylabel('Loss', color='b')
        ax2.legend(loc=4)
        #plt.ylim([0.0, 2.5])
        plt.show()

    def predict(self, sample_dataset, ds_info):
        predictions = self.model.predict(sample_dataset)
        class_names = ds_info.features['label'].names

        # Print the predicted labels for the sample
        for i, prediction in enumerate(predictions):
            predicted_label = tf.argmax(prediction).numpy()
            print(f"Sample {i+1}: Predicted label - {class_names[predicted_label]}")

    def plot_predictions(self, sample_dataset, ds_info, num_samples=25, grid_shape=(5, 5)):  
        data_size = tf.data.experimental.cardinality(sample_dataset)  
        assert data_size == grid_shape[0] * grid_shape[1], "The sample count should match the grid plot count."    

        images = []
        true_labels = []
        for data in sample_dataset:
            image, label = data
            images.append(image)
            true_labels.append(label)

        predictions = self.model.predict(sample_dataset.batch(num_samples))

        class_names = ds_info.features['label'].names
        fig, axes = plt.subplots(*grid_shape, figsize=(5, 5))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            image = images[i]
            true_label = true_labels[i].numpy()
            predicted_label = np.argmax(predictions[i])
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'True: {class_names[true_label]}\nPred: {class_names[predicted_label]}', fontsize=8)
        plt.tight_layout()
        plt.show()

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()