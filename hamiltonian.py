import numpy as np
import scipy as sc
from scipy import sparse
from scipy.linalg import expm


class RydbergHamiltonian1D:
    def __init__(self, n, a=1, C_6=1, δ=1, rabi=1):
        '''

        :param n:
        :param a:
        :param C_6:
        :param δ:
        :param rabi:
        '''
        self.n = n
        self.a = a
        self.C_6 = C_6
        self.δ = δ
        self.rabi = rabi
        self.id = np.identity(2)
        self.σx = np.array([[0, 1], [1, 0]])
        self.dimension = 2 ** n
        self.zeros = np.zeros((self.dimension, self.dimension))

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
        m = self.tensor_prod_id(i, np.array([[0, 0], [0, 1]]))

        return m

    def sum_n_i(self):
        m = self.zeros

        for j in range(1, self.n + 1):
            m = m + self.n_i(j)

        return m

    def vdw(self):

        m_vdw = self.zeros

        for i in range(1, self.n + 1):
            for k in range(1, i):
                r = self.a * abs(i - k)
                v = self.C_6 / r ** 6
                m_ik = v * np.dot(self.n_i(i), self.n_i(k))
                m_vdw = m_vdw + m_ik

        return m_vdw

    def hamiltonian_matrix(self, δ):
        h_m = ((self.rabi / 2) * self.sum_sigma_xi()) - (δ * self.sum_n_i()) + self.vdw()

        return h_m


class AdiabaticEvolution(RydbergHamiltonian1D):
    def __init__(self, n, t, dt):
        super().__init__(n)
        self.t = t
        self.dt = dt
        self.steps = int(t / dt)
        self.detunning = np.linspace(-1, 1, self.steps)

    def ground_state(self):
        g = np.zeros((2 ** self.n, 1))
        g[0, 0] = 1

        return g

    def time_evolve(self):
        ψ = self.ground_state()
        j = self.basis_vectors(2**(self.n-1))
        for i in range(0, self.steps):
            ψ = np.dot(expm(-1j * super().hamiltonian_matrix(self.detunning[i]) * self.dt), ψ)
            density_matrix = np.dot(ψ, ψ.T)
        for i in range(0,self.n):
            j_bra = j[i]
            j_ket = j[i].T
            print(j_ket)
            m_left = np.kron(self.id, j_bra)
            m_right = np.kron(self.id, j_ket)
            density_matrix_one = np.dot(m_left, np.dot(density_matrix, m_right))

        return density_matrix_one

    @staticmethod
    def basis_vectors(n):
        return [np.array([np.eye(n)[i]]) for i in range(n)]

    def reduced_density_matrix(self, i):
        j_bra = [np.array([[1, 0]]), np.array([[0, 1]])]
        j_ket = [np.array([[1, 0]]).T, np.array([[0, 1]]).T]

        m = [1]


        for j in range(0,2**(self.n-1)):



            m_ll = self.basis_vectors(i)[j]
            m_lr = self.basis_vectors(self.n-i-1)[j]






if __name__ == "__main__":
    #two = RydbergHamiltonian1D(2)

    dt = 0.1
    t = 10

    #evol = AdiabaticEvolution(3, t, dt)

    #v = evol.time_evolve()


