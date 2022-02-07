from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.io import mmread
from scipy.io import mmwrite
from scipy.linalg import sqrtm
from skimage import transform
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import sys

def initial(path, downfactor):
    """
    initialize the sparse matrix
    input: path of mtx sparse file
    downfactor: down sampling factor
    """
    df = pd.read_csv(path)
    row = df['user'].values
    col = df['item'].values
    rate = df['rating'].values

    m = max(row) + 1 # row
    n = max(col) + 1 # col

    sparsecsr = csr_matrix((rate, (row, col)), shape=(m, n))

    shape = sparsecsr.shape
    m_ = round(m/downfactor)
    n_ = round(n/downfactor)
    nnz = sparsecsr.nnz
    print(f'Read sparse matrix row: {m}, col: {n}, nnz: {nnz}')
    k = min(m_, n_)
    print(f'down factor is {downfactor}')
    return sparsecsr, k, m_, n_

def image_resize(mat, m, n):
    """
    use skimage to resize the matrix
    """
    return transform.resize(mat, (m, n))

def compute_reduce(sparse, k, m, n):
    """
    compute the svd of sparse matrix
    then apply the frobenius norm
    """
    print("start svd...")
    u, s, v = svds(sparse, k)
    print("image resizing...")
    u_ = np.array(image_resize(u, k, n))
    v_ = np.array(image_resize(v, m, k))
    u_t = u_.transpose()
    v_t = v_.transpose()
    norm_u = u_ @ sqrtm(u_t @ u_)
    norm_v = sqrtm(v_ @ v_t)  @ v_
    r_tmp = norm_u @ np.diag(s) @ norm_v

    M = np.max(np.max(r_tmp))
    m = np.min(np.min(r_tmp))
    reduced = r_tmp / (M-m)
    return reduced

def fractal_expansion(spmat, reduced):
    """
    use kronecker product
    """
    return sparse.kron(spmat, reduced)

if __name__ == '__main__':
    sparsepath = sys.argv[1] # output.csv
    downfactor = int(sys.argv[2])
    spmat, k, m_, n_ = initial(sparsepath, downfactor)
    reduced = compute_reduce(spmat, k, m_, n_)
    out = fractal_expansion(spmat, reduced)
    print("start writing")
    mmwrite('out.mtx', out)
