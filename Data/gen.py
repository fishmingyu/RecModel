from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.io import mmread
from scipy.io import mmwrite
from scipy.linalg import sqrtm
from skimage import transform
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
import sys

class graphExpansion(object):
    def __init__(self, nodes_pair, downfactor) -> None:
        """
        initialize the sparse matrix
        input: path of mtx sparse file
        downfactor: down sampling factor
        """
        self.nodes_pair = nodes_pair
        self.downfactor = downfactor
        self.row = nodes_pair['user'].values
        self.col = nodes_pair['item'].values
        self.rate = np.random.randn(len(self.row))

        self.m = max(self.row) + 1 # row
        self.n = max(self.col) + 1 # col
        self.nnz = len(self.row) # nnz
        
        self.spmat = coo_matrix((self.rate, (self.row, self.col)), shape=(self.m, self.n))

        self.m_down = round(self.m/downfactor)
        self.n_down = round(self.n/downfactor)
        
        print(f'Read sparse matrix row: {self.m}, col: {self.n}, nnz: {self.nnz}')
        self.k = min(self.m_down, self.n_down)
        print(f'down factor is {downfactor}')

    def image_resize(self, mat, m, n):
        """
        use skimage to resize the matrix
        """
        return transform.resize(mat, (m, n))

    def compute_reduce(self, sparse, k, m, n):
        """
        compute the svd of sparse matrix
        then apply the frobenius norm
        """
        print("start svd...")
        u, s, v = svds(sparse, k)
        print("image resizing...")
        u_ = np.array(self.image_resize(u, n, k))
        v_ = np.array(self.image_resize(v, k, m))
        u_t = u_.transpose()
        v_t = v_.transpose()
        norm_u = u_ @ sqrtm(u_t @ u_)
        norm_v = sqrtm(v_ @ v_t)  @ v_

        r_tmp = norm_u @ np.diag(s) @ norm_v

        M = np.max(np.max(r_tmp))
        m = np.min(np.min(r_tmp))
        reduced = r_tmp / (M-m)
        return reduced

    def process(self):
        """
        output a csr_matrix in scipy
        """
        if self.downfactor == 1:
            out = self.fractal_expansion(self.spmat, self.spmat)
        else:
            reduced = self.compute_reduce(self.spmat, self.k, self.m_down, self.n_down)
            out = self.fractal_expansion(self.spmat, reduced)
        out = out.tocoo()
        row = out.row
        col = out.col
        
        data_dict = {'user':row, 'item':col}

        return pd.DataFrame(data_dict)

    def fractal_expansion(self, spmat, reduced):
        """
        use kronecker product
        """
        return sparse.kron(spmat, reduced)

