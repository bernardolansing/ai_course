import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

cifar10 = keras.datasets.cifar10

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def get_cifar10_network():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), # Camada de entrada (conv)
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(),                            # Flatten antes das densas
        keras.layers.Dense(128, activation='relu'),        # 1ª camada densa: 128 neurônios
        keras.layers.Dense(64, activation='relu'),         # 2ª camada densa: 64 neurônios
        keras.layers.Dense(10, activation='softmax')       # Camada de saída: 10 neurônios
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = get_cifar10_network()
    
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
