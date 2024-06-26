import numpy as np
import scipy as sc
from scipy import sparse
from scipy.linalg import expm
from matplotlib import pyplot as plt
from scipy.constants import Planck

import data_analysis


class RydbergHamiltonian1D:
    def __init__(self, n, a=5.48, C_6= 862690*2 * np.pi, Rabi= 4*2 * np.pi, NN=False):
        '''

        :param n: Number of atoms
        :param a: Separation between atoms (μm)
        :param C_6: Interaction strength (2π x Mhz μm^6)
        :param Rabi: Rabi frequency (2π x Mhz)
        '''
        self.n = n
        self.a = a
        self.C_6 = C_6
        self.Rabi = Rabi
        self.id = np.identity(2)
        self.σx = np.array([[0, 1], [1, 0]])
        self.ni_op = np.array([[0, 0], [0, 1]])
        self.dimension = 2 ** n
        self.zeros = np.zeros((self.dimension, self.dimension))
        self.h = Planck * 1e6
        self.h_bar = self.h / (2 * np.pi)
        self.two_pi = 2 * np.pi
        self.NN = NN

        if self.Rabi == 0:
            self.r_b = 0
        else:
            self.r_b = (C_6 / Rabi) ** (1 / 6)

    def tensor_prod_id(self, i, matrix):
        m0 = [1]
        m1 = matrix

        for j in range(0, i - 1):
            m0 = sparse.kron(m0, self.id).toarray()

        for k in range(i, self.n):
            m1 = sparse.kron(m1, self.id).toarray()

        M = sparse.kron(m0, m1).toarray()

        return M

    def sigma_xi(self, i):
        m = self.tensor_prod_id(i, self.σx)

        return m

    def sum_sigma_xi(self):
        m = self.zeros

        for j in range(1, self.n + 1):
            m = m + self.sigma_xi(j)

        return m

    def n_i(self, i):
        m = self.tensor_prod_id(i, self.ni_op)

        return m

    def sum_n_i(self, δ=None):
        m = self.zeros

        if δ is None:
            for j in range(1, self.n + 1):
                m = m + self.n_i(j)

        else:
            for j in range(1, self.n + 1):
                m = m + (δ[j - 1] * self.n_i(j))

        return m

    def vdw(self, first_atom_move=0):

        m_vdw = self.zeros

        if self.C_6 != 0:

            for i in range(1, self.n + 1):
                for k in range(1, i):
                    if k == 1:
                        r = self.a * (abs(i - k) + first_atom_move)
                    else:
                        r = self.a * (abs(i - k))
                        
                    if self.NN:
                        if i-k == 1:
                            v = self.C_6 / r ** 6
    
                        else:
                            v=0

                    else:
                        v = self.C_6 / r ** 6

                    m_ik = v * np.dot(self.n_i(i), self.n_i(k))
                    m_vdw = m_vdw + m_ik

        return m_vdw

    def hamiltonian_matrix(self, δ, first_atom_move=None, rabi_regime=1):

        if len(δ) > 1:
            if first_atom_move is None:
                h_m = (((rabi_regime * self.Rabi / 2) * self.sum_sigma_xi()) - self.sum_n_i(δ=δ) + self.vdw())
            else:
                h_m = (((rabi_regime * self.Rabi / 2) * self.sum_sigma_xi()) - self.sum_n_i(δ=δ) + self.vdw(first_atom_move=first_atom_move))


        else:
            if first_atom_move is None:
                h_m = (((rabi_regime * self.Rabi / 2) * self.sum_sigma_xi()) - δ*self.sum_n_i() + self.vdw())
            else:
                h_m = (((rabi_regime * self.Rabi / 2) * self.sum_sigma_xi()) - δ*self.sum_n_i() + self.vdw(first_atom_move=first_atom_move))

        return h_m


if __name__ == "__main__":

    h_m = RydbergHamiltonian1D(3).hamiltonian_matrix([0])
    h_m = h_m / (2 * np.pi)

    A = [[0,1,1,0],
         ]

    eigenvalues, eigenvector = np.linalg.eigh(h_m)

    print(h_m)

    print(eigenvalues)
    print(eigenvector)

    #print(data_analysis.thermalization_matrix(h_m))








