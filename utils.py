import numpy as np

def normalize_mat(matrix):
    for i in range(len(matrix)):
        matrix[i:] /= matrix[i:].sum()

    return matrix
