import numpy as np

def horizontal_shear(s):
    return np.array([[1.0, s], [0.0, 1.0]])

def vertical_shear(s):
    return np.array([[1.0, 0.0], [s, 1.0]])

def compose_maps(*matrices):
    result = np.eye(2)
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("Each matrix must have shape (2, 2).")
        result = matrix @ result
    return result

def apply_map(points, matrix, periodic=False):
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    if points.ndim not in (1, 2) or points.shape[-1] != 2:
        raise ValueError("Points must have shape (2,) or (N, 2).")
    if matrix.shape != (2, 2):
        raise ValueError("Matrix must have shape (2, 2).")
    if periodic:
        if not np.allclose(matrix, np.rint(matrix)) or not np.isclose(np.linalg.det(matrix), 1):
            raise ValueError("Periodic maps require integer matrices with determinant 1.")

    transformed = points @ matrix.T
    return np.mod(transformed, 1.0) if periodic else transformed
