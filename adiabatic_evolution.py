import sys
import numpy as np
import scipy as sc
from scipy import sparse
from scipy.linalg import expm
from scipy.linalg import logm
from matplotlib import pyplot as plt
import math


from rydberg_hamiltonian_1d import RydbergHamiltonian1D

import detuning_regimes
import data_analysis
import data_analysis as da
import rabi_regimes



class AdiabaticEvolution(RydbergHamiltonian1D):
    def __init__(self, n, t, dt, δ_start, δ_end, a=5.48, rabi_osc=False, no_int=False, detuning_type='linear',
                 single_addressing_list=None, initial_state_list=None, rabi_regime="constant", Rabi= 4*2 * np.pi, NN=False):
        super().__init__(n, a=a, Rabi=Rabi, NN=NN)
        self.t = t
        self.dt = dt
        self.steps = int(t / dt)
        self.δ_start = δ_start
        self.δ_end = δ_end
        self.times = np.arange(0, t, self.dt)
        self.reduced_den_initial = np.zeros((2, 2))

        if rabi_osc:
            self.detunning = np.zeros(self.steps)

        if no_int:
            self.C_6 = 0

        # Detuning Regime

        if single_addressing_list is not None:

            self.detunning = detuning_regimes.single_addressing(self.t, self.dt, self.δ_start, self.δ_end,
                                                                single_addressing_list)

        else:
            self.detunning = detuning_regimes.global_detuning(t, dt, δ_start, δ_end, d_type=detuning_type)

        # Rabi regime
        self.rabi_regime = rabi_regimes.global_rabi(self.t, self.dt, self.steps, type=rabi_regime)

        # Initial regime
        if initial_state_list is not None:
            # if len(initial_state_list) == self.n:
            self.initial_psi = self.initial_state(initial_state_list)
            # else:
            #     sys.exit()

        else:
            self.initial_psi = self.ground_state()

    def ground_state(self):
        g = np.zeros((2 ** self.n, 1))
        g[0, 0] = 1

        return g

    def bel_psi_minus(self):
        v = np.zeros((4, 1))
        v[1, 0] = 1 / (2 ** 0.5)
        v[2, 0] = -1 / (2 ** 0.5)

        return v

    def bel_psi_plus(self):
        v = np.zeros((4, 1))
        v[1, 0] = 1 / (2 ** 0.5)
        v[2, 0] = 1 / (2 ** 0.5)

        return v

    def initial_state(self, state_list, bell=False):
        state_list = list(reversed(state_list))
        if not bell:
            v = self.col_basis_vectors(2)[state_list[0]]
            start = 1
        else:
            v = self.bel_psi_plus()
            start=2

        for i in state_list[start:]:
            u = self.col_basis_vectors(2)[i]
            v = np.kron(v, u) # Order Switched

        return v

    def time_evolve(self, density_matrix=False, rydberg_fidelity=False, bell_fidelity=False, bell_fidelity_types=None,
                    states_list=False, reduced_density_matrix_pair=False, hms=False, expec_energy=False,
                    eigen_list=False, eigenstate_fidelities=False, rabi_regime_type='constant',
                    entanglement_entropy=False, std_energy_list=False):

        ψ = self.initial_psi
        j = self.row_basis_vectors(2 ** (self.n - 1))
        rydberg_fidelity_list = [[] for _ in range(self.n)]
        density_matrices = []
        hamiltonian_matrices = []
        eigenvalues = []
        eigenvectors = []
        expectation_energies = []
        eigenstate_probs = [[] for _ in range(self.dimension)]
        bell_fidelity_list = [[] for _ in range(self.n - 1)]
        bell_fidelity_dict = {'psi plus': [[] for _ in range(self.n - 1)], 'psi minus': [[] for _ in range(self.n - 1)],
                              'phi plus': [[] for _ in range(self.n - 1)], 'phi minus': [[] for _ in range(self.n - 1)]}
        states = []
        rdms_pairs_list = [[] for _ in range(self.n - 1)]
        entanglement_entropy_list = []
        std_energies = []

        # Moving the first atom
        first_atom_move = np.linspace(0, 0, self.steps) #np.hstack((np.linspace(1000,0, int(self.steps/2)), np.linspace(0,0, int(self.steps/2))))

        #np.hstack((np.linspace(0,0, int(self.steps/2)), np.linspace(0,10, int(self.steps/2))))

        for k in range(0, self.steps):

            h_m = self.hamiltonian_matrix(self.detunning[:, k], first_atom_move=first_atom_move[k], rabi_regime=self.rabi_regime[k])
            ψ = np.dot(expm(-1j * h_m * self.dt), ψ)

            if density_matrix:
                density_m = np.dot(ψ, ψ.conj().T)
                density_matrices += [density_m]
                if expec_energy:
                    ee = data_analysis.expectation_value(density_m, h_m)
                    expectation_energies += [ee]

            elif expec_energy:
                density_m = np.dot(ψ, ψ.conj().T)
                ee = data_analysis.expectation_value(density_m, h_m)
                expectation_energies += [ee]

            if std_energy_list:
                density_m = np.dot(ψ, ψ.conj().T)
                std = data_analysis.std_observable(density_m, h_m)
                std_energies += [std]


            if eigen_list:
                eigenvalue, eigenvector = np.linalg.eigh(h_m)
                eigenvalues += [eigenvalue]
                eigenvectors += [eigenvector]
                if eigenstate_fidelities:
                    for i in range(0, self.dimension):
                        v = eigenvector[:, i]
                        v = v[:, np.newaxis]
                        eigenstate_prob = data_analysis.state_prob(v, ψ)
                        eigenstate_probs[i] += [eigenstate_prob]

            if entanglement_entropy:
                rdm = self.reduced_density_matrix_half(ψ)
                print(np.shape(rdm))

                vne = self.entanglement_entropy(rdm)

                entanglement_entropy_list += [vne]

            if hms:
                hamiltonian_matrices += [h_m]

            if rydberg_fidelity:
                self.rydberg_fidelity(rydberg_fidelity_list, ψ)

            if states_list:
                states += [ψ]

            if bell_fidelity:
                self.bell_state_fidelity(bell_fidelity_list, ψ)

            if bell_fidelity_types is not None:
                for i in bell_fidelity_types:
                    self.bell_state_fidelity(bell_fidelity_dict[i], ψ, bell_type=i)

            if reduced_density_matrix_pair:
                for i in range(1, self.n):
                    rdm = self.reduced_density_matrix_pair(ψ, i, i + 1)
                    rdms_pairs_list[i - 1] += [rdm]

        if rydberg_fidelity:
            if states_list:
                if entanglement_entropy:
                    return rydberg_fidelity_list, states, entanglement_entropy_list
                else:
                    return rydberg_fidelity_list, states
            elif bell_fidelity_types:
                return rydberg_fidelity_list, bell_fidelity_dict
            elif bell_fidelity:
                return rydberg_fidelity_list, bell_fidelity_list
            elif hms:
                return rydberg_fidelity_list, hamiltonian_matrices
            elif eigen_list:
                if expec_energy and eigenstate_fidelities:
                    return rydberg_fidelity_list, eigenvalues, eigenvectors, expectation_energies, eigenstate_probs
                elif expec_energy:
                    return rydberg_fidelity_list, eigenvalues, eigenvectors, expectation_energies
                elif eigenstate_fidelities:
                    return rydberg_fidelity_list, eigenvalues, eigenvectors, eigenstate_probs
            elif density_matrix:
                return rydberg_fidelity_list, density_matrices
            elif entanglement_entropy:
                return rydberg_fidelity_list, entanglement_entropy_list
            else:
                return rydberg_fidelity_list

        elif density_matrix:
            if hms:
                return density_matrices, hamiltonian_matrices
            elif expec_energy:
                return density_matrices, expectation_energies
            else:
                return density_matrices

        elif eigen_list:
            if expec_energy and eigenstate_fidelities:
                if states_list:
                    return eigenvalues, eigenvectors, expectation_energies, eigenstate_probs, states
                elif std_energy_list:
                    return eigenvalues, eigenvectors, expectation_energies, eigenstate_probs, std_energies
                else:
                    return eigenvalues, eigenvectors, expectation_energies, eigenstate_probs
            elif expec_energy:
                return eigenvalues, eigenvectors, expectation_energies
            elif eigenstate_fidelities:
                return eigenvalues, eigenvectors, eigenstate_probs
            elif states_list:
                return eigenvalues, eigenvectors, states
            else:
                return eigenvalues, eigenvectors

        elif expec_energy:
            if std_energy_list:
                return expectation_energies, std_energies

        elif hms:
            return hamiltonian_matrices

        elif states_list:
            if entanglement_entropy:
                return states, entanglement_entropy_list
            else:
                return states

        elif bell_fidelity:
            return bell_fidelity_list

        elif bell_fidelity_types:
            return bell_fidelity_dict

        elif reduced_density_matrix_pair:
            return rdms_pairs_list

        elif entanglement_entropy:
            return entanglement_entropy_list

        elif std_energy_list:
            return std_energies

        elif eigenstate_fidelities:
            return eigenstate_probs

        else:
            return ψ

    @staticmethod
    def comp_basis_vector_to_qubit_states(basis_vector):

        rows = np.shape(basis_vector)[0]
        cols = np.shape(basis_vector)[1]

        # Case for bra vector
        if cols > rows:
            n = int(np.log2(cols))

            # Convert the basis vector to binary representation
            basis_int = int(math.log2(int(''.join(map(str, reversed(basis_vector[0]))), 2)))
            binary_rep = format(basis_int, f'0{n}b')

            # Initialize a list to store individual qubit states
            qubit_states = []

            # Split the binary representation into n parts
            for i in range(n):
                qubit_binary = binary_rep[i]  # Get the binary value for the i-th qubit
                qubit_states.append(np.array([1 - int(qubit_binary), int(qubit_binary)]))  # Convert to 2D state vector

        # Case for ket vector
        elif rows > cols:
            n = int(np.log2(rows))

            # Convert the basis vector to binary representation
            basis_int = int(math.log2(int(''.join(map(str, reversed(basis_vector[:, 0]))), 2)))
            binary_rep = format(basis_int, f'0{n}b')

            # Initialize a list to store individual qubit states
            qubit_states = []

            # Split the binary representation into n parts
            for i in range(n):
                qubit_binary = binary_rep[i]  # Get the binary value for the i-th qubit
                qubit_states += [np.array([[1 - int(qubit_binary), int(qubit_binary)]]).T]  # Convert to 2D state vector

        else:
            raise ValueError('rows and cols equal')
            sys.exit()

        return qubit_states

    @staticmethod
    def row_basis_vectors(n):
        return [np.array([np.eye(n)[i]]).astype(int) for i in range(n)]

    @staticmethod
    def col_basis_vectors(n):
        return [np.array([np.eye(n)[i]]).astype(int).T for i in range(n)]

    def reduced_density_matrix(self, ψ, i):

        if self.n == 1:
            reduced_density_matrix = np.dot(ψ, ψ.conj().T)

        else:
            dim_sub = 2 ** (self.n - 1)

            m_left_list = []
            m_right_list = []

            # Produce list of matrices on left side of sum
            row_basis_vectors = self.row_basis_vectors(dim_sub)
            for row_vector in row_basis_vectors:
                bra_vectors = self.comp_basis_vector_to_qubit_states(row_vector)
                bra_vectors.insert(i - 1, self.id)

                m_left = bra_vectors[0]  # initialise left matrix

                for bra in bra_vectors[1:]:
                    m_left = np.kron(m_left, bra)  # taking tensor product left to right

                m_left_list += [m_left]

            # Produce list of matrices on right side of sum
            col_basis_vectors = self.col_basis_vectors(dim_sub)
            for col_vector in col_basis_vectors:
                ket_vectors = self.comp_basis_vector_to_qubit_states(col_vector)
                ket_vectors.insert(i - 1, self.id)

                m_right = ket_vectors[0]  # initialise right matrix

                for ket in ket_vectors[1:]:
                    m_right = np.kron(m_right, ket)  # taking tensor product left to right

                m_right_list += [m_right]

            reduced_density_matrix = np.zeros((2, 2))

            for j in range(0, dim_sub):
                m_left = m_left_list[j]
                m_right = m_right_list[j]
                reduced_density_matrix = reduced_density_matrix + np.dot(np.dot(m_left, ψ), np.dot(ψ.conj().T, m_right))

        return reduced_density_matrix

    def reduced_density_matrix_pair(self, ψ, i, j):

        if self.n == 1:
            sys.exit()

        dim_sub = 2 ** (self.n - 2)

        m_left_list = []
        m_right_list = []

        # Produce list of matrices on left side of sum
        row_basis_vectors = self.row_basis_vectors(dim_sub)
        for row_vector in row_basis_vectors:
            bra_vectors = self.comp_basis_vector_to_qubit_states(row_vector)
            bra_vectors.insert(i - 1, self.id)
            bra_vectors.insert(j - 1, self.id)

            m_left = bra_vectors[0]  # initialise left matrix

            for bra in bra_vectors[1:]:
                m_left = np.kron(m_left, bra)  # taking tensor product left to right

            m_left_list += [m_left]

        # Produce list of matrices on right side of sum
        col_basis_vectors = self.col_basis_vectors(dim_sub)
        for col_vector in col_basis_vectors:
            ket_vectors = self.comp_basis_vector_to_qubit_states(col_vector)
            ket_vectors.insert(i - 1, self.id)
            ket_vectors.insert(j - 1, self.id)

            m_right = ket_vectors[0]  # initialise right matrix

            for ket in ket_vectors[1:]:
                m_right = np.kron(m_right, ket)  # taking tensor product left to right

            m_right_list += [m_right]

        reduced_density_matrix = np.zeros((4, 4))

        for j in range(0, dim_sub):
            m_left = m_left_list[j]
            m_right = m_right_list[j]
            reduced_density_matrix = reduced_density_matrix + np.dot(np.dot(m_left, ψ), np.dot(ψ.conj().T, m_right))

        return reduced_density_matrix

    def reduced_density_matrix_half(self, ψ):

        if self.n == 1:
            sys.exit()

        n_A = math.ceil(self.n / 2)
        n_B = self.n - n_A

        d_A = 2 ** n_A
        d_B = 2 ** n_B

        m_left_list = []
        m_right_list = []

        # Produce list of matrices on left side of sum
        row_basis_vectors = self.row_basis_vectors(d_B)
        for row_vector in row_basis_vectors:
            bra_vectors = self.comp_basis_vector_to_qubit_states(row_vector)

            m_left = np.eye(d_A)  # initialise left matrix

            for bra in bra_vectors:
                m_left = np.kron(m_left, bra)  # taking tensor product left to right
            # m_left = np.eye(d_A)
            # m_left = np.kron(m_left, row_vector)

            m_left_list += [m_left]

        # Produce list of matrices on right side of sum
        col_basis_vectors = self.col_basis_vectors(d_B)
        for col_vector in col_basis_vectors:
            ket_vectors = self.comp_basis_vector_to_qubit_states(col_vector)

            m_right = np.eye(d_A) # initialise right matrix

            for ket in ket_vectors:
                m_right = np.kron(m_right, ket)  # taking tensor product left to right

            m_right_list += [m_right]

        reduced_density_matrix = np.zeros((d_A, d_A))

        for j in range(0, d_B):
            m_left = m_left_list[j]
            m_right = m_right_list[j]
            reduced_density_matrix = reduced_density_matrix + np.dot(np.dot(m_left, ψ), np.dot(ψ.conj().T, m_right))

        return reduced_density_matrix

    def reduced_density_matrix_from_left(self, n, n_A , ψ):

        if n == 1:
            sys.exit()

        n_B = n - n_A

        d_A = 2 ** n_A
        d_B = 2 ** n_B

        m_left_list = []
        m_right_list = []

        # Produce list of matrices on left side of sum
        row_basis_vectors = self.row_basis_vectors(d_B)
        for row_vector in row_basis_vectors:
            bra_vectors = self.comp_basis_vector_to_qubit_states(row_vector)

            m_left = np.eye(d_A)  # initialise left matrix

            for bra in bra_vectors:
                m_left = np.kron(m_left, bra)  # taking tensor product left to right
            # m_left = np.eye(d_A)
            # m_left = np.kron(m_left, row_vector)

            m_left_list += [m_left]

        # Produce list of matrices on right side of sum
        col_basis_vectors = self.col_basis_vectors(d_B)
        for col_vector in col_basis_vectors:
            ket_vectors = self.comp_basis_vector_to_qubit_states(col_vector)

            m_right = np.eye(d_A) # initialise right matrix

            for ket in ket_vectors:
                m_right = np.kron(m_right, ket)  # taking tensor product left to right

            m_right_list += [m_right]

        reduced_density_matrix = np.zeros((d_A, d_A))

        for j in range(0, d_B):
            m_left = m_left_list[j]
            m_right = m_right_list[j]
            reduced_density_matrix = reduced_density_matrix + np.dot(np.dot(m_left, ψ), np.dot(ψ.conj().T, m_right))

        return reduced_density_matrix

    def rydberg_fidelity(self, rydberg_fidelity_list, ψ):

        for i in range(1, self.n + 1):
            rdm = self.reduced_density_matrix(ψ, i)
            rf = data_analysis.expectation_value(rdm, self.ni_op)
            rydberg_fidelity_list[i - 1] += [rf]

    def bell_state_fidelity(self, bell_fidelity_list, ψ, bell_type='psi minus'):

        if bell_type == 'psi plus':
            bell_state = 0.5 * np.array([[0, 0, 0, 0],
                                         [0, 1, 1, 0],
                                         [0, 1, 1, 0],
                                         [0, 0, 0, 0]])


        elif bell_type == 'psi minus':
            bell_state = 0.5 * np.array([[0, 0, 0, 0],
                                         [0, 1, -1, 0],
                                         [0, -1, 1, 0],
                                         [0, 0, 0, 0]])

        elif bell_type == 'phi plus':
            bell_state = 0.5 * np.array([[1, 0, 0, 1],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [1, 0, 0, 1]])

        elif bell_type == 'phi minus':
            bell_state = 0.5 * np.array([[1, 0, 0, -1],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [-1, 0, 0, 1]])

        else:
            sys.exit()

        for i in range(1, self.n):
            rdm = self.reduced_density_matrix_pair(ψ, i, i + 1)
            bf = data_analysis.expectation_value(rdm, bell_state)
            bell_fidelity_list[i - 1] += [bf]

    def entanglement_entropy(self, rdm):

        vne = data_analysis.von_nuemann_entropy(rdm)

        return vne

    def rydberg_rydberg_density_corr_function(self, psi, i=None):

        g = [0]*(self.n-1)

        if i is None:
            print('yes')

            for r in range(1, self.n):
                for k in range(1, self.n+1-r):
                    g[r-1] = g[r-1] + data_analysis.correlation_funtction(self.reduced_density_matrix_pair(psi, k, k+r), self.reduced_density_matrix(psi, k), self.reduced_density_matrix(psi, k+r))

                g[r-1] = g[r-1]/(self.n-r)

        elif i == 'pair':
            for k in range(2, self.n+1):
                j=k-1
                g[k-2] = data_analysis.correlation_funtction(self.reduced_density_matrix_pair(psi, j, k), self.reduced_density_matrix(psi, j), self.reduced_density_matrix(psi, k))

        else:
            for i in range(2, self.n+1):
                g[i-2] = data_analysis.correlation_funtction(self.reduced_density_matrix_pair(psi, 1, i), self.reduced_density_matrix(psi, 1), self.reduced_density_matrix(psi, i))

        return g

    def concurrence(self, psi, c_type='pair'):

        if c_type =='pair':
            C = [0]*(self.n-1)
            for k in range(2, self.n+1):
                j=k-1
                print(j)
                C[k-2] = data_analysis.concurrence(self.reduced_density_matrix_pair(psi,j,k))

        elif type(c_type) == int:
            C = [0] * (self.n - 1)
            for k in range(1, self.n + 1):
                if k > c_type:
                    C[k - 2] = data_analysis.concurrence(self.reduced_density_matrix_pair(psi, c_type, k))
                elif k < c_type:
                    C[k - 1] = data_analysis.concurrence(self.reduced_density_matrix_pair(psi, k, c_type))
                else:
                    pass

        elif c_type == 'both sides':
            C = [0] * int((self.n-1)/2)

            for k in range(1, int((self.n+1)/2)):
                j = self.n+1-k
                C[k - 1] = data_analysis.concurrence(self.reduced_density_matrix_pair(psi, k, j))


            print(C)

        else:
            sys.exit()



        return C











if __name__ == "__main__":
    t = 6
    dt = 0.01
    n = 3
    δ_start = -200
    δ_end = 200

    evol = AdiabaticEvolution(n, t, dt, δ_start, δ_end, detuning_type='linear')

    #psi = evol.time_evolve(states_list=True)

    psi = 1/(np.sqrt(2)) * np.array([[0.9], [np.sqrt(0.195)], [0], [0], [0], [0], [0], [0.9]])
    #psi = 1/(np.sqrt(2)) * np.array([[1], [0], [0], [1]])

    rdm = evol.reduced_density_matrix_pair(psi,1,2)

    C = data_analysis.concurrence(rdm)

    print(evol.initial_state([0,0,1]))

    print(rdm)
    print(C)







    # rdms = evol.time_evolve(reduced_density_matrix_pair=True)

    # v = evol.bel_psi_minus()

