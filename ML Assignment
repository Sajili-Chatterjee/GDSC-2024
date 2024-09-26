import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class CustomActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.k0 = tf.Variable(0.0, trainable=True)
        self.k1 = tf.Variable(1.0, trainable=True)

    def call(self, inputs):
        return self.k0 + self.k1 * inputs

# The neural network model
def create_model(input_shape):
    mnist_model = Sequential()
    mnist_model.add(Dense(16, input_shape=(input_shape,), activation=CustomActivation()))
    mnist_model.add(Dense(10, activation='softmax'))
    return mnist_model

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

# Training the model
def train_model(mnist_model, x_train, y_train, x_test, y_test):
    mnist_model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    history = mnist_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    return history

def plot_results(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    mnist_model = create_model(input_shape=28 * 28)
    history = train_model(mnist_model, x_train, y_train, x_test, y_test)

    plot_results(history)
