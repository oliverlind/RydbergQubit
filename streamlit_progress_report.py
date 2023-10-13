import streamlit as st
from PIL import Image

def main_ui():
    """
    Main streamlit UI
    :return:
    """

    # Page config
    st.set_page_config(
        page_title="Oliver Lind Rydberg Atom Quantum Computing",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Title and introduction
    st.title('Oliver Lind Rydberg Atom Quantum Computing')
    st.write('Drag and drop files in sidebar to produce plots and report')

    with st.sidebar:
        st.selectbox('Select Week', ['Intro', 'Week 1'])

    st.code('''class RydbergHamiltonian1D:
    def __init__(self, n, a=1, C_6=1, δ=1, rabi=1):
        
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

        return h_m''', language="python")

    image = Image.open('Plots/Figure_1.png')

    st.image(image)

if __name__ == "__main__":
    main_ui()