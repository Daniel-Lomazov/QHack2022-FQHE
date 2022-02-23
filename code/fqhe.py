from typing import Optional

import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True


def verify_nij(n_blocks, fqhe_circuit):
    """
    Verifies eq. 16 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """
    n_t = 13
    tvals = np.round(np.linspace(0, 1.2, n_t), 3)

    t_output = []

    for t in tvals:
        # ni = fqhe_circuit(n_blocks, measure_ni(n_blocks), t)
        n0i = fqhe_circuit(n_blocks, measure_nij, t)

        twopt = []
        for j in range(3 * n_blocks - 1):
            ni = n0i[j][0] + n0i[j][1]
            nj = n0i[j][2] + n0i[j][3]

            nij = n0i[-1]
            twopt.append(nij - ni * nj)
        t_output.append(twopt)

    fig, ax = plt.subplots()
    im = ax.imshow(t_output)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(n_t), labels=tvals)
    ax.set_ylabel("t", rotation=0)

    ax.set_xticks(np.arange(3 * n_blocks - 1))
    ax.set_xlabel(r'$i - j$')
    ax.set_title(
        r'$\textit{2 point correlation function} \; \left|\left\langle n_{i}n_{j}\right\rangle -\left\langle n_{i}\right\rangle \left\langle n_{j}\right\rangle \right|$')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        r'$\left|\left\langle n_{i}n_{j}\right\rangle -\left\langle n_{i}\right\rangle \left\langle n_{j}\right\rangle \right|$',
        rotation=90, va="bottom")
    plt.show()

    return


def verify_ni(n_blocks, fqhe_circuit):
    """
    Verifies eq. 15 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """

    n_t = 13
    tvals = np.round(np.linspace(0, 1.2, n_t), 3)
    n = [fqhe_circuit(n_blocks, measure_ni(n_blocks), phi(t, n_blocks)) for t in tvals]

    fig, ax = plt.subplots()
    im = ax.imshow(n)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(n_t), labels=tvals)
    ax.set_ylabel("t", rotation=0)

    ax.set_xticks(np.arange(3 * n_blocks))
    ax.set_xlabel(r'$\left\langle n_i \right\rangle$')
    ax.set_title(r'$\left\langle n_i \right\rangle \; \textit{as function of t}$')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r'$\left\langle n_i \right\rangle$', rotation=-0, va="bottom")
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


def red_ij(red_wires):
    def ret(n_blocks):
        return qml.density_matrix(wires=red_wires)

    return ret


def measure_ni(n_blocks):
    def ret():
        obs = [qml.Hermitian(0.5 * (qml.Identity(i).matrix + qml.PauliZ(i).matrix), wires=i) for i in
               range(3 * n_blocks)]
        return [qml.expval(o) for o in obs]

    return ret


def phi(t: float, n_blocks: int) -> list[float]:
    """
    This function computes the phases for the terms in the v=1/3 FQHE quantum state.

    Parameters
    ----------
    t: Potential strength t=sqrt(V_30/V_10)
    n_blocks: Number of 3-qubit blocks in the FQHE system

    Returns
    -------
    Angles phi_i for the bozonized system, that ensure correct phases
    """
    phi_i = [np.arctan(-t)]
    for i in range(n_blocks - 1):
        phi_i.insert(0, np.arctan(-t * np.cos(phi_i[0])))

    return phi_i


def fqhe_circuit(n_blocks, obs, phi_i: list[float]) -> list[float]:
    """

    Parameters
    ----------
    n_blocks: Number of 3-qubit blocks in the FQHE system.
    t: Potential strength t=sqrt(V_30/V_10)
    phi_i: Angles phi_i

    Returns
    -------
    The v=1/3 FQHE state

    """
    if phi_i is None:
        raise ValueError("Must provide phi")

    # Stage 0
    for i in range(n_blocks + 1):
        qml.PauliX(3 * i)

    # Stage 1
    qml.RY(-2 * phi_i[0], wires=[1])
    for i in range(n_blocks - 1):
        qml.CRY(-2 * phi_i[i + 1], wires=[3 * i + 1, 3 * (i + 1) + 1])

    # Stage 2 - part 1
    for i in range(n_blocks):
        qml.CNOT(wires=[3 * i + 1, 3 * i + 2])

    # Stage 2 - part 2
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=3 * i + 1)
        qml.CNOT(wires=[3 * i + 2, 3 * (i + 1)])

    # Stage 2 - part 3
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=3 * i + 2)
        qml.CNOT(wires=[3 * i + 1, 3 * i])

    return obs()


if __name__ == '__main__':
    n_blocks = 7
    n_wires = 3 * n_blocks + 2
    # dev1 = qml.device("default.qubit", wires=n_wires, shots=60)
    # fqhe = qml.QNode(fqhe_circuit, dev1)

    my_bucket = "amazon-braket-amazon-braket-47ba137bf31d"  # the name of the bucket, keep the 'amazon-braket-' prefix and then include the bucket name
    my_prefix = "penny"  # the name of the folder in the bucket
    s3_folder = (my_bucket, my_prefix)

    device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    dev_remote = qml.device('braket.aws.qubit', device_arn=device_arn, wires=n_wires, shots=10)
    fqhe_remote = qml.QNode(fqhe_circuit, dev_remote)

    verify_ni(n_blocks, fqhe_remote)
    # verify_nij(n_blocks, fqhe)
