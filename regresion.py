import numpy as np
from numpy._typing import NDArray
from typing import Tuple

def definir_variables(x: NDArray) -> Tuple[NDArray[np.float32], float]:
    assert(len(x.shape) == 2), 'Not valid array'  ## (n_datos, n_features) -> (n_datos,n_features) * (n_features,) -> n_datos, n_features (broadcasting)
    return np.random.normal(size = (x.shape[-1],)).astype(np.float32), np.random.normal(size = 1).astype(np.float32)

def dJ_base(x: NDArray, w: NDArray, b: float, y: NDArray) -> NDArray:
    out = (x@w) + b - y
    return out

def dJ_dwn(x: NDArray, n: int, base: NDArray) -> float:
    return 0.5 * (base*x[:, n]).mean()

def dJ_db(base: NDArray) -> float:
    return 0.5 * base.mean()

def J(base) -> float:
    return (base**2).mean()

def paso_gradiente(x: NDArray, w: NDArray, b: float, alpha: float, base: NDArray) -> Tuple[NDArray[np.float32], float]:
    for n in range(len(w)):
        w[n] -= alpha*dJ_dwn(x, n, base)
    b -= alpha*dJ_db(base)
    return w, b

def regresion(epochs: int, alpha: float, x: NDArray, y: NDArray) -> None:
    w, b = definir_variables(x)
    for epoch in range(epochs):
        base = dJ_base(x, w, b, y)
        w, b = paso_gradiente(x, w, b, alpha, base)
        costo = J(base)
        print(f'Epoca {epoch}: El costo es {costo}')

    out = 'Modelo final: f(x) = '
    for i, w_n in enumerate(w):
        out +=f'{w_n:.4f}x_{i} + '
    print(out + f'{b[0]:.4f}')

if __name__ == '__main__':
    epochs: int = 30
    alpha: float = 1e-2
    x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ], dtype = np.float32)

    y = np.array([10,26,42], dtype = np.float32)

    regresion(epochs, alpha, x, y)
