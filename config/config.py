
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

    if n == 4:
        legend_labels = ["$E_{0}$", "$E_{1}$", "$E_{2}$", "$E_{3}$", "$E_{4}$", "$E_{5}$", "$E_{6}$", "$E_{7}$", '⟨E⟩']*2
        return legend_labels

plotcolors = [
    'blue', 'green', 'red', 'cyan', 'magenta',
    'orange', 'purple', 'pink', 'brown', 'darkgreen', 'skyblue', 'salmon',
    'grey', 'olive', 'orchid', 'teal', 'gold'
]

if __name__ == "__main__":
    print(energy_eigenvalue_labels(3))
