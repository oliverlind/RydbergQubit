import numpy as np
import scipy as sc
from scipy import sparse
from scipy.linalg import expm
from matplotlib import pyplot as plt


class RydbergHamiltonian1D:
    def __init__(self, n, a=1, C_6=0, δ=0, rabi=1):
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
        self.ni_op = np.array([[0, 0], [0, 1]])
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
        m = self.tensor_prod_id(i, self.ni_op)

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
    def __init__(self, n, t, dt, δ_start=-1, δ_end=1, rabi_osc=False):
        super().__init__(n)
        self.t = t
        self.dt = dt
        self.steps = int(t / dt)
        self.detunning = np.linspace(δ_start, δ_end, self.steps)
        self.times = np.linspace(0, t, self.steps)
        self.reduced_den_initial = np.zeros((2, 2))

        if rabi_osc:
            self.detunning = np.zeros(self.steps)

    def ground_state(self):
        g = np.zeros((2 ** self.n, 1))
        g[0, 0] = 1

        return g

    def time_evolve(self, density_matrix=False, rydberg_fidelity=False):
        ψ = self.ground_state()
        j = self.row_basis_vectors(2 ** (self.n - 1))
        rydberg_fidelity = {}

        for k in range(0, self.steps):
            ψ = np.dot(expm(-1j * self.hamiltonian_matrix(self.detunning[k]) * self.dt), ψ)

            if density_matrix:
                density_matrix = np.dot(ψ, ψ.conj().T)

            if rydberg_fidelity:
                pass



            # if self.n == 2:
            #     rydberg_fidelity = self.two_atom_evolve(density_matrix, j)
            #
            # if self.n == 3:
            #     rydberg_fidelity = self.three_atom_evolve(density_matrix, j)

        rdm = self.reduced_density_matrix(ψ, 2)
        print(rdm)

        # plt.plot(self.times, rydberg_fidelity['Atom 1'], label='Atom 1')
        # plt.plot(self.times, rydberg_fidelity['Atom 2'], label='Atom 2')
        # plt.legend(loc='upper right')
        #
        # plt.show()

    def two_atom_evolve(self, density_matrix, j, rydberg_fidelity={}, n_1list=[], n_2list=[]):

        # Initialise reduced density matrices to 0
        density_matrix_one = self.reduced_den_initial
        density_matrix_two = self.reduced_den_initial

        # Calculate reduced density matrices for each atom
        for i in range(0, 2**(self.n-1)):
            j_bra = j[i]
            j_ket = j[i].conj().T

            # Atom 1
            m_left = np.kron(self.id, j_bra)
            m_right = np.kron(self.id, j_ket)
            density_matrix_one = density_matrix_one + np.dot(m_left, np.dot(density_matrix, m_right))

            # Atom 2
            m_left = np.kron(j_bra, self.id)
            m_right = np.kron(j_ket, self.id)
            density_matrix_two = density_matrix_two + np.dot(m_left, np.dot(density_matrix, m_right))

        # Rydberg Fidelity Atom 1
        n_1 = abs(np.trace(np.dot(density_matrix_one, self.ni_op)))
        n_1list += [n_1]

        # Rydberg Fidelity Atom 2
        n_2 = abs(np.trace(np.dot(density_matrix_two, self.ni_op)))
        n_2list += [n_2]

        rydberg_fidelity['Atom 1'] = n_1list
        rydberg_fidelity['Atom 2'] = n_2list

        return rydberg_fidelity

    def three_atom_evolve(self, density_matrix, j, rydberg_fidelity={}, n_1list=[], n_2list=[], n_3list=[]):

        # Initialise reduced density matrices to 0
        density_matrix_one = self.reduced_den_initial
        density_matrix_two = self.reduced_den_initial
        density_matrix_three = self.reduced_den_initial

        # Calculate reduced density matrices for each atom
        for i in range(0, 2 ** (self.n - 1)):
            j_bra = j[i]
            j_ket = j[i].conj().T

            # Atom one
            m_left = np.kron(self.id, j_bra)
            m_right = np.kron(self.id, j_ket)
            density_matrix_one = density_matrix_one + np.dot(m_left, np.dot(density_matrix, m_right))

            # Atom two
            self.comp_basis_vector_to_qubit_states(j_bra)


            # Atom three
            m_left = np.kron(j_bra, self.id)
            m_right = np.kron(j_ket, self.id)
            density_matrix_two = density_matrix_two + np.dot(m_left, np.dot(density_matrix, m_right))

        # Rydberg Fidelity Atom 1
        n_1 = abs(np.trace(np.dot(density_matrix_one, self.ni_op)))
        n_1list += [n_1]

        # Rydberg Fidelity Atom 2
        n_2 = abs(np.trace(np.dot(density_matrix_two, self.ni_op)))
        n_2list += [n_2]

        rydberg_fidelity['Atom 1'] = n_1list
        rydberg_fidelity['Atom 2'] = n_2list

        return rydberg_fidelity

    @staticmethod
    def comp_basis_vector_to_qubit_states(basis_vector):

        rows = np.shape(basis_vector)[0]
        cols = np.shape(basis_vector)[1]

        # Case for bra vector
        if cols > rows:
            n = int(np.log2(cols))

            # Convert the basis vector to binary representation
            basis_int = int(np.log2(int(''.join(map(str, reversed(basis_vector[0]))), 2)))
            binary_rep = format(basis_int, f'0{n}b')

            # Initialize a list to store individual qubit states
            qubit_states = []

            # Split the binary representation into n parts
            for i in range(n):
                qubit_binary = binary_rep[i]  # Get the binary value for the i-th qubit
                qubit_states.append(np.array([1 - int(qubit_binary), int(qubit_binary)]))  # Convert to 2D state vector

        # Case for ket vector
        if rows > cols:
            n = int(np.log2(rows))

            # Convert the basis vector to binary representation
            basis_int = int(np.log2(int(''.join(map(str, reversed(basis_vector[:, 0]))), 2)))
            binary_rep = format(basis_int, f'0{n}b')

            # Initialize a list to store individual qubit states
            qubit_states = []

            # Split the binary representation into n parts
            for i in range(n):
                qubit_binary = binary_rep[i]  # Get the binary value for the i-th qubit
                qubit_states += [np.array([[1 - int(qubit_binary), int(qubit_binary)]]).T]  # Convert to 2D state vector

        return qubit_states

    @staticmethod
    def row_basis_vectors(n):
        return [np.array([np.eye(n)[i]]).astype(int) for i in range(n)]

    @staticmethod
    def col_basis_vectors(n):
        return [np.array([np.eye(n)[i]]).astype(int).T for i in range(n)]

    def reduced_density_matrix(self, ψ, i):
        dim_sub = 2 ** (self.n - 1)

        m_left_list = []
        m_right_list = []

        # Produce list of matrices on left side of sum
        row_basis_vectors = self.row_basis_vectors(dim_sub)
        for row_vector in row_basis_vectors:
            bra_vectors = self.comp_basis_vector_to_qubit_states(row_vector)
            bra_vectors.insert(i-1, self.id)

            m_left = bra_vectors[0]  # initialise left matrix

            for bra in bra_vectors[1:]:
                m_left = np.kron(m_left, bra)  # taking tensor product left to right

            m_left_list += [m_left]

        # Produce list of matrices on right side of sum
        col_basis_vectors = self.col_basis_vectors(dim_sub)
        for col_vector in col_basis_vectors:
            ket_vectors = self.comp_basis_vector_to_qubit_states(col_vector)
            ket_vectors.insert(i-1, self.id)

            m_right = ket_vectors[0]  # initialise right matrix

            for ket in ket_vectors[1:]:
                m_right = np.kron(m_right, ket)  # taking tensor product left to right

            m_right_list += [m_right]

        reduced_density_matrix = np.zeros((dim_sub,dim_sub))

        for j in range(0, dim_sub):
            m_left = m_left_list[j]
            m_right = m_right_list[j]
            reduced_density_matrix = reduced_density_matrix + np.dot(np.dot(m_left, ψ), np.dot(ψ.conj().T, m_right))

        return reduced_density_matrix

    def rydberg_fidelity(self,ψ):


        for i in range(1, self.n+1):
            rdm = self.reduced_density_matrix(ψ, i)
            rf = np.trace(np.dot(rdm, self.ni_op))










if __name__ == "__main__":
    # two = RydbergHamiltonian1D(2)
    # #
    # h_m = two.hamiltonian_matrix(0)
    # #
    # # eigenvalues = np.linalg.eigvals(h_m)
    # #
    # print(h_m)


    #
    # # print(h_m)
    #
    dt = 0.1
    t = 30
    # #
    evol = AdiabaticEvolution(2, t, dt, rabi_osc=True)
    # #rydberg = evol.time_evolve()

    basis_vector = np.array([[0, 0, 0, 1, 0, 0, 0, 0]])

    evol.time_evolve()














