import tensorflow as tf
from tensorflow import keras
from livelossplot import PlotLossesKeras
import time

fashion_mnist = keras.datasets.fashion_mnist


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def get_fashion_mnist_network():
    model = keras.Sequential([
        tf.keras.layers.Conv2D(24, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=1),

        keras.layers.Flatten(),  # Camada de entrada (flatten)
        keras.layers.Dense(512, activation='relu'),  # 1ª camada: 512 neurônios
        keras.layers.Dense(256, activation='relu'),  # 2ª camada: 256 neurônios
        keras.layers.Dense(128, activation='relu'),  # 3ª camada: 128 neurônios
        keras.layers.Dense(64, activation='relu'),  # 4ª camada: 64 neurônios
        keras.layers.Dense(32, activation='relu'),  # 5ª camada: 32 neurônios
        keras.layers.Dense(10, activation='softmax')  # Saída: 10 classes
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = get_fashion_mnist_network()

    start_time = time.time()

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[PlotLossesKeras()])

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\n--- Resultados do Treinamento ---")
    print(f"Tempo de treinamento: {training_time:.2f} segundos ({training_time / 60:.2f} minutos)")

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")


if __name__ == "__main__":
    run()
