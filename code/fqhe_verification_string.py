import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import strftime
from fqhe_prep import phi

plt.rcParams['text.usetex'] = True

def measure_string_ij(i, j):
    """
    Based on calculation in "calculating_string_operators.lyx"
    Parameters
    ----------
    i
    j

    Returns
    -------

    """

    def ret():
        str_ij = qml.PauliZ(wires=3 * i + 6) @ qml.PauliZ(wires=3 * i + 4)
        for k in range(i + 1, j - 1):
            str_ij = str_ij @ qml.PauliZ(wires=3 * k + 6) @ qml.PauliZ(wires=3 * k + 4)

        o1 = [qml.PauliZ(wires=3 * i + 3) @ str_ij @ qml.PauliZ(wires=3 * j + 3), qml.PauliZ(
            wires=3 * i + 3) @ str_ij, str_ij @ qml.PauliZ(wires=3 * j + 3), str_ij]

        o2 = [-1 * qml.PauliZ(wires=3 * i + 3) @ str_ij @ qml.PauliZ(wires=3 * j + 1), -1 * qml.PauliZ(
            wires=3 * i + 3) @ str_ij, -1 * str_ij @ qml.PauliZ(wires=3 * j + 1), -1 * str_ij]

        o3 = [-1 * qml.PauliZ(wires=3 * i + 1) @ str_ij @ qml.PauliZ(wires=3 * j + 3), -1 * qml.PauliZ(
            wires=3 * i + 1) @ str_ij, -1 * str_ij @ qml.PauliZ(wires=3 * j + 3), -1 * str_ij]

        o4 = [qml.PauliZ(wires=3 * i + 1) @ str_ij @ qml.PauliZ(wires=3 * j + 1), qml.PauliZ(
            wires=3 * i + 1) @ str_ij, str_ij @ qml.PauliZ(wires=3 * j + 1), str_ij]

        obs = []
        for o in [o1, o2, o3, o4]:
            obs.extend(o)
        return [qml.expval(o) for o in obs]

    return ret


def verify_string_data(n_blocks, fqhe_circuit):
    ivals = [1, 2, 3, 4, 5]
    t = 1
    phi_i = phi(t, n_blocks)

    jrange = len(range(1, 3 * n_blocks))
    str_output = np.zeros((len(ivals), jrange))
    for idx, i in tqdm(enumerate(ivals)):
        string_ij = [fqhe_circuit(n_blocks, measure_string_ij(i, j), phi_i) for j in
                     range(i + 1, 3 * n_blocks)]
        string_ij = [string_ij[0:4], string_ij[4:8], string_ij[8:12], string_ij[12:16]]
        str_output[idx, len(range(1, 3 * n_blocks)) - len(string_ij):] = [s[0] - s[1] - s[2] + s[3] for s in string_ij]

    return ivals, str_output


def verify_string(n_blocks, fqhe_circuit, n_shots):
    """
    Verifies eq. 16 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """
    tvals, t_output = verify_string_data(n_blocks, fqhe_circuit)

    fig, ax = plt.subplots()
    im = ax.imshow(t_output)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(tvals)), labels=tvals)
    ax.set_ylabel("t", rotation=0)

    ax.set_xticks(np.arange(3 * n_blocks - 1))
    ax.set_xlabel(r'$i - j$')
    ax.set_title(r'$O^{ij}_str\; \textit{string operator order parameter,\#shots= \;' +
                 str(n_shots) + r', \#N= ' + str(3 * n_blocks) + r'}$')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r'$O^{ij}_str$', rotation=90, va="top")
    plt.show()

    return