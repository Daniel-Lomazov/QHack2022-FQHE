import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from fqhe import phi


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


def measure_nij(n_blocks, i=0):
    def ret():
        return [qml.probs(wires=[i, j]) for j in range(1, 3 * n_blocks)]

    return ret
    # coeffs = np.ones(4) / 4
    # obs = []
    # for j in range(1, 3 * n_blocks):
    #     obs = [qml.Identity(i) @ qml.Identity(j), qml.PauliZ(i) @ qml.Identity(j), qml.PauliZ(j) @ qml.Identity(i),
    #            qml.PauliZ(i) @ qml.PauliZ(j)]
    #     Hj = qml.Hamiltonian(coeffs, obs)
    #     obs.append((Hj))

    # obs = [qml.PauliZ(j) for j in range(3 * n_blocks)]
    # obs.extend([qml.PauliZ(i) @ qml.PauliZ(j) for j in range(1, 3 * n_blocks)])
    # return obs

    # opi = qml.Hermitian(0.25 * (qml.Identity(i).matrix + qml.PauliZ(i).matrix), wires=0).matrix
    # obs = [qml.Hermitian(np.kron(opi, (qml.Identity(j).matrix + qml.PauliZ(j).matrix)), wires=[i, j]) for j in
    #        range(1, 3 * n_blocks)]
    #
    # return obs


def verify_nij_data(n_blocks, fqhe_circuit):
    n_t = 13
    tvals = np.round(np.linspace(0, 1.2, n_t), 3)

    t_output = []

    for t in tvals:
        # ni = fqhe_circuit(n_blocks, measure_ni(n_blocks), t)
        n0i = fqhe_circuit(n_blocks, measure_nij, phi(t, n_blocks))

        twopt = []
        for j in range(3 * n_blocks - 1):
            ni = n0i[j][0] + n0i[j][1]
            nj = n0i[j][2] + n0i[j][3]

            nij = n0i[-1]
            twopt.append(nij - ni * nj)
        t_output.append(twopt)

    return tvals, t_output


def verify_nij(n_blocks, fqhe_circuit, n_shots):
    """
    Verifies eq. 16 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """
    tvals, t_output = verify_nij_data(n_blocks, fqhe_circuit, n_shots)

    fig, ax = plt.subplots()
    im = ax.imshow(t_output)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(tvals)), labels=tvals)
    ax.set_ylabel("t", rotation=0)

    ax.set_xticks(np.arange(3 * n_blocks - 1))
    ax.set_xlabel(r'$i - j$')
    ax.set_title(
        r'$\textit{2 point correlation function} \;'
        r' \left|\left\langle n_{i}n_{j}\right\rangle -\left\langle n_{i}\right\rangle \left\langle n_{j}\right\rangle \right|; \#shots=' + str(
            n_shots) + r'}$')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        r'$\left|\left\langle n_{i}n_{j}\right\rangle -\left\langle n_{i}\right\rangle \left\langle n_{j}\right\rangle \right|$',
        rotation=90, va="bottom")
    plt.show()

    return


def verify_ni(n_blocks, fqhe_circuit, n_shots):
    """
    Verifies eq. 15 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """

    n_t = 12
    tvals = np.round(np.linspace(0, 1.2, n_t), 3)
    typename = type(fqhe_circuit.device).__dict__['name']
    measure = measure_ni(n_blocks, typename=typename)
    n = [np.array(fqhe_circuit(n_blocks, measure, phi(t, n_blocks))) for t in tqdm(tvals)]

    if "Aws" in typename:
        n = [2 * o - 1 for o in n]

    n_new = []
    for idx in range(len(n)):
        n_new.append(n[-1 - idx])

    fig, ax = plt.subplots()
    im = ax.imshow(n_new)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(-0.5, -0.5 + n_t), labels=np.flip(tvals))
    ax.set_ylabel("t", rotation=0)

    ax.set_xticks(np.arange(-0.5, -0.5 + 3 * (n_blocks + 1)), labels=range(3 * (n_blocks + 1)))
    ax.set_xlabel(r'$\textit{\large{k}}$')
    ax.set_title('N = {}; shots = {}'.format(3 * (n_blocks + 1), n_shots))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r'$\left\langle n_k \right\rangle$', rotation=90, va="bottom")
    plt.show()

    return


def measure_ni(n_blocks, typename=""):
    def ret():
        if "Aws" in typename:
            obs = [qml.PauliZ(wires=i) for i in range(3 * (n_blocks + 1))]
        else:
            obs = [qml.Hermitian(0.5 * (qml.Identity(wires=i).matrix - qml.PauliZ(wires=i).matrix), wires=i) for i in
                   range(3 * (n_blocks + 1))]
        return [qml.expval(o) for o in obs]

    return ret
