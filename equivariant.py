import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Input is shaped:
# Batch x N x N x L
# Where each L indexes a permutation
# invariant 2D tensor
# All the __ funs are tests, implemented slowly (but correctly)

def diag_to_diag_1(input):
    diag = tf.einsum("...iil ->...li", input)
    return tf.einsum("...lij->...ijl", tf.linalg.diag(diag))

def __diag_to_diag_1(input):
    res = np.zeros_like(input)
    numnonzero = 0


    b, n, _, l = res.shape
    for i in range(b):
        for j in range(n):
            for k in range(l):
                res[i][j][j][k] = input[i][j][j][k]
                numnonzero += 1

    analytic_nonzero = b * n * l # Diagonal in each batch and l
    assert(numnonzero == analytic_nonzero)

    return res, numnonzero

def diag_to_rows_2(input):
    diag = tf.einsum("...iil ->...li", input)
    return tf.einsum("...li, ...lj ->...ijl", tf.ones_like(diag), diag)


def __diag_to_rows_2(input):
    res = np.zeros_like(input)
    numnonzero = 0
    b, n, _, L = res.shape

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = input[batch][j][j][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)
    return res, numnonzero

def diag_to_cols_3(input):
    diag = tf.einsum("...iil ->...li", input)
    return tf.einsum("...li, ...lj->...jil", tf.ones_like(diag), diag)


def __diag_to_cols_3(input):
    res = np.zeros_like(input)
    numnonzero = 0
    b, n, _, L = res.shape

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = input[batch][i][i][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)
    return res, numnonzero

def rowsum_to_diag_4(input):
    rowsum = tf.einsum("...ijl -> ...li", input)
    return tf.einsum("...lij->...ijl", tf.linalg.diag(rowsum))


def __rowsum_to_diag_4(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, l = res.shape
    rowsums = np.zeros(
        shape=(b, n, l)
    )

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for k in range(l):
                    rowsums[batch][i][k] += input[batch][i][j][k]

    for i in range(b):
        for j in range(n):
            for k in range(l):
                res[i][j][j][k] = rowsums[i][j][k]
                numnonzero += 1

    analytical_nonzero = b * n * l
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero


def colsum_to_diag_5(input):
    colsum = tf.einsum("...ijl -> ...lj", input)
    return tf.einsum("...lij->...ijl", tf.linalg.diag(colsum))


def __colsum_to_diag_5(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, l = res.shape
    colsums = np.zeros(
        shape=(b, n, l)
    )

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for k in range(l):
                    colsums[batch][i][k] += input[batch][j][i][k]

    for i in range(b):
        for j in range(n):
            for k in range(l):
                res[i][j][j][k] = colsums[i][j][k]
                numnonzero += 1

    analytical_nonzero = b * n * l
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero

def trace_to_all_6(input):
    trace = tf.einsum("...iil->...l", input)
    return tf.einsum("...ijl, ...l->...ijl", tf.ones_like(input), trace)

def __trace_to_all_6(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, L = res.shape
    traces = np.zeros(
        shape=(b, L)
    )

    for batch in range(b):
        for i in range(n):
            for l in range(L):
                traces[batch][l] += input[batch][i][i][l]

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = traces[batch][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero

def transpose_to_all_7(input):
    return tf.einsum("...ijl->...jil", input)

def __transpose_to_all_7(input):
    res = np.zeros_like(input)
    numnonzero = 0
    b, n, _, L = res.shape

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = input[batch][j][i][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero


def trace_to_diag_9(input):
    trace = tf.einsum("...iil->...l", input)
    diag = tf.einsum("...iil ->...li", input)
    A = tf.eye(num_rows=diag.shape[-1], batch_shape=trace.shape)
    return tf.einsum("...l, ...lij->...ijl", trace, A)


def __trace_to_diag_9(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, l = res.shape
    traces = np.zeros(
        shape=(b, l)
    )

    for batch in range(b):
        for i in range(n):
            for j in range(l):
                traces[batch][j] += input[batch][i][i][j]

    for batch in range(b):
        for i in range(n):
            for j in range(l):
                res[batch][i][i][j] = traces[batch][j]
                numnonzero += 1

    analytical_nonzero = b * n * l
    assert(numnonzero==analytical_nonzero)

    return res, numnonzero

def rowsum_to_cols_10(input):
    rowsum = tf.einsum("...ijl -> ...li", input)
    return tf.einsum("...li, ...lj->...jil", tf.ones_like(rowsum), rowsum)

def __rowsum_to_cols_10(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, L = res.shape

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = np.sum(input[batch, i, :, l])
                    # for k in range(n):
                    #     res[batch][i][j][l] += input[batch][i][k][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero

def colsum_to_cols_11(input):
    colsum = tf.einsum("...ijl -> ...lj", input)
    return tf.einsum("...li, ...lj->...jil", tf.ones_like(colsum), colsum)

def __colsum_to_cols_11(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, L = res.shape

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = np.sum(input[batch, :, i, l])
                    # for k in range(n):
                    #     res[batch][i][j][l] += input[batch][i][k][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero

def totsum_to_diag_12(input):
    totsum = tf.einsum("...ijl -> ...l", input)
    trace = tf.einsum("...iil->...l", input)
    diag = tf.einsum("...iil ->...li", input)
    A = tf.eye(num_rows=diag.shape[-1], batch_shape=trace.shape)
    return tf.einsum("...l, ...lij->...ijl", totsum, A)


def __totsum_to_diag_12(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, l = res.shape
    totsums = np.zeros(
        shape=(b, l)
    )

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for k in range(l):
                    totsums[batch][k] += input[batch][i][j][k]


    for batch in range(b):
        for i in range(n):
            for j in range(l):
                res[batch][i][i][j] = totsums[batch][j]
                numnonzero += 1

    analytical_nonzero = b * n * l
    assert(numnonzero==analytical_nonzero)

    return res, numnonzero

def colsum_to_rows_13(input):
    colsum = tf.einsum("...ijl -> ...lj", input)
    return tf.einsum("...li, ...lj->...ijl", tf.ones_like(colsum), colsum)

def __colsum_to_rows_13(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, L = res.shape

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = np.sum(input[batch, :, j, l])

                    # for k in range(n):
                    #     res[batch][i][j][l] += input[batch][k][j][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero


def rowsum_to_rows_14(input):
    rowsum = tf.einsum("...ijl -> ...li", input)
    return tf.einsum("...li, ...lj->...ijl", tf.ones_like(rowsum), rowsum)

def __rowsum_to_rows_14(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, L = res.shape

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = np.sum(input[batch, j, :, l])
                    # for k in range(n):
                    #     res[batch][i][j][l] += input[batch][i][k][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero

def totsum_to_all_15(input):
    totsum = tf.einsum("...ijl -> ...l", input)
    return tf.einsum("...ijl, ...l->...ijl", tf.ones_like(input), totsum)

def __totsum_to_all_15(input):
    res = np.zeros_like(input)
    numnonzero = 0

    b, n, _, L = res.shape
    totsums = np.zeros(
        shape=(b, L)
    )

    for batch in range(b):
        for l in range(L):
            totsums[batch][l] = np.sum(input[batch, :, :, l])

    for batch in range(b):
        for i in range(n):
            for j in range(n):
                for l in range(L):
                    res[batch][i][j][l] = totsums[batch][l]
                    numnonzero += 1

    analytical_nonzero = b * n * n * L
    assert(numnonzero == analytical_nonzero)

    return res, numnonzero




def test_all(eps=1e-6):

    B = np.random.randint(10, 20)
    N =  np.random.randint(5, 10)
    L =  np.random.randint(5, 10)

    funcs = [
        (diag_to_diag_1, __diag_to_diag_1),
        (diag_to_rows_2, __diag_to_rows_2),
        (diag_to_cols_3, __diag_to_cols_3),
        (rowsum_to_diag_4, __rowsum_to_diag_4),
        (colsum_to_diag_5, __colsum_to_diag_5),
        (trace_to_all_6, __trace_to_all_6),
        (transpose_to_all_7, __transpose_to_all_7),
        (trace_to_diag_9, __trace_to_diag_9),
        (rowsum_to_cols_10, __rowsum_to_cols_10),
        (colsum_to_cols_11, __colsum_to_cols_11),
        (totsum_to_diag_12, __totsum_to_diag_12),
        (colsum_to_rows_13, __colsum_to_rows_13),
        (rowsum_to_rows_14, __rowsum_to_rows_14),
        (totsum_to_all_15, __totsum_to_all_15)
    ]



    tensor = tf.random.uniform(
        shape=(B, N, N, L)
    )

    with tqdm(total=len(funcs)) as pbar:
        for func, test in funcs:
            pbar.set_description(f"Func: {func.__name__}")
            func_val = func(tensor)
            test_val, num_nonzero = test(tensor)

            diff = tf.linalg.norm(
                func_val - test_val
            ) / num_nonzero
            # Want the average norm to be less than eps
            # where the average is over all nonzero elements

            if diff > eps:
                raise RuntimeError(f"Function {func.__name__} does not match its test to precision {eps} < {diff}")

            pbar.update(1)

    print("All tests passed")











