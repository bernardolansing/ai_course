import tensorflow as tf
from tensorflow import keras
from livelossplot import PlotLossesKeras
import time

cifar100 = keras.datasets.cifar100


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def get_cifar100_network():
    model = keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(100, activation='softmax')  # Camada de saída: 100 classes
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = get_cifar100_network()

    start_time = time.time()

    # early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=5,
    #                                                  restore_best_weights=True)
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[PlotLossesKeras()])

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\n--- Resultados do Treinamento ---")
    print(f"Tempo de treinamento: {training_time:.2f} segundos ({training_time / 60:.2f} minutos)")

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")


if __name__ == "__main__":
    run()
