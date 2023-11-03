
# Legend labels
def energy_eigenvalue_labels(n):

    if n == 1:
        legend_labels = ["$E_{0}$", "$E_{1}$", '⟨E⟩']
        return legend_labels

    if n ==2:
        legend_labels = ["$E_{0}$", "$E_{1}$", "$E_{2}$", "$E_{3}$", '⟨E⟩']
        return legend_labels

    if n == 3:
        legend_labels = ["$E_{0}$", "$E_{1}$", "$E_{2}$", "$E_{3}$", "$E_{4}$", "$E_{5}$", "$E_{6}$", "$E_{7}$", '⟨E⟩']
        return legend_labels

if __name__ == "__main__":
    print(energy_eigenvalue_labels(3))