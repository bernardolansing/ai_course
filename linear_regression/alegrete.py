import numpy as np
from sklearn.metrics import mean_squared_error


def compute_mse(b, w, data):
    """
    Calcula o erro quadratico medio
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """
    N = len(data)
    X = data[:, 0]
    Y = data[:, 1]
    Y_pred = w * X + b
    mse = (1 / N) * np.sum((Y - Y_pred) ** 2)
    return mse


def step_gradient(b, w, data, alpha):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de b e w.
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de b e w, respectivamente
    """
    N = len(data)
    x = data[:, 0]
    y = data[:, 1]
    y_pred = b + w * x
    b_gradient = (-2 / N) * sum(y - y_pred)
    w_gradient = (-2 / N) * sum(x * (y - y_pred))
    new_b = b - alpha * b_gradient
    new_w = w - alpha * w_gradient
    return new_b, new_w


def fit(data, b, w, alpha, num_iterations):
    """
    Para cada época/iteração, executa uma atualização por descida de
    gradiente e registra os valores atualizados de b e w.
    Ao final, retorna duas listas, uma com os b e outra com os w
    obtidos ao longo da execução (o último valor das listas deve
    corresponder à última época/iteração).

    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :param num_iterations: int - numero de épocas/iterações para executar a descida de gradiente
    :return: list,list - uma lista com os b e outra com os w obtidos ao longo da execução
    """
    b_history = [b]
    w_history = [w]
    for i in range(num_iterations):
        b, w = step_gradient(b, w, data, alpha)
        b_history.append(b)
        w_history.append(w)
    return b_history, w_history

def run():
    # Carrega o dataset
    dataset = np.loadtxt('./datasets/alegrete.csv', delimiter=',')
    alpha = 0.01
    num_iterations = 1000
    initial_b = 0.0
    initial_w = 0.0
    b_history, w_history = fit(dataset, initial_b, initial_w, alpha, num_iterations)
    print("Final b:", b_history[-1])
    print("Final w:", w_history[-1])
    print("Final MSE:", compute_mse(b_history[-1], w_history[-1], dataset))

if __name__ == "__main__":
    run()

