import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data():
    """
    Loads the CIFAR10 dataset and normalizes pixel values to be between 0 and 1.
    """
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def create_model():
    """
    Creates a Convolutional Neural Network (CNN) for image classification.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model

def train_and_evaluate(model, train_data, test_data, epochs=10):
    """
    Compiles, trains, and evaluates the model.
    """
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels))
    
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')
    return history

def plot_history(history):
    """
    Plots training and validation accuracy/loss over epochs.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.show()

if __name__ == "__main__":
    # Class names for CIFAR10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Execution pipeline
    train_data, test_data = load_and_preprocess_data()
    model = create_model()
    model.summary()
    history = train_and_evaluate(model, train_data, test_data)
    # plot_history(history) # Uncomment in a local environment with GUI support