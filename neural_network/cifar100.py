import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

cifar100 = keras.datasets.cifar100

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def get_cifar100_network():
    model = keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),#(32, 32, 3) porque as imagens são 32X32 e RGB, portanto, tendo 3 canais de cor
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(100, activation='softmax')      # Camada de saída: 100 classes
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = get_cifar100_network()
    
    start_time = time.time()
    
    model.fit(x_train, y_train, epochs=10)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n--- Resultados do Treinamento ---")
    print(f"Tempo de treinamento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    run()
