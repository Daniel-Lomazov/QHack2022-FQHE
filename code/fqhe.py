from typing import Optional
import argparse
from tqdm import tqdm  # For daniel: pip install tqdm
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

plt.rcParams['text.usetex'] = True


def verify_nij_data(n_blocks, fqhe_circuit):
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
    measure = aws_measure_ni(n_blocks)
    n = [fqhe_circuit(n_blocks, measure, phi(t, n_blocks)) for t in tqdm(tvals)]

    fig, ax = plt.subplots()
    im = ax.imshow(n)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(n_t), labels=tvals)
    ax.set_ylabel("t", rotation=0)

    ax.set_xticks(np.arange(3 * n_blocks))
    ax.set_xlabel(r'$\left\langle n_i \right\rangle$')
    ax.set_title(r'$\left\langle n_i \right\rangle \; \textit{as function of t; \#shots=' + str(n_shots) + r'}$')

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
    """
    Example for using red_ij on first block:
        fqhe_circuit(5, red_ij([0,1,2]), phi_i)
    Parameters
    ----------
    red_wires

    Returns
    -------

    """

    def ret():
        return qml.density_matrix(wires=red_wires)

    return ret


def measure_ni(n_blocks):
    def ret():
        obs = [qml.Hermitian(0.5 * (qml.Identity(i).matrix + qml.PauliZ(i).matrix), wires=i) for i in
               range(3 * n_blocks)]
        return [qml.expval(o) for o in obs]

    return ret


def measure_string_ij(n_block, i, j):
    def ret():
        None

    return None


def aws_measure_ni(n_blocks):
    def ret():
        obs = [qml.Hermitian(0.5 * ([[1, 0], [0, 1]] + qml.PauliZ(i).matrix), wires=i) for i in
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


def local_device(n_wires, n_shots):
    dev = qml.device("default.qubit", wires=n_wires, shots=n_shots)
    return dev


def braket_device(n_wires, n_shots):
    my_bucket = "amazon-braket-amazon-braket-47ba137bf31d"
    my_prefix = "penny"
    s3_folder = (my_bucket, my_prefix)

    sv1_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    # aspen11_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-11"
    dev_remote = qml.device('braket.aws.qubit', device_arn=sv1_arn, wires=n_wires, shots=n_shots)
    return dev_remote


def get_circuit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="local")
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--shots', type=int, default=10)
    args = parser.parse_args()
    n_blocks = args.blocks
    n_shots = args.shots
    dev_name = args.device

    print("Runningg FQHE v=1/3 simulation with device-{} on {} blocks for {} shots".format(dev_name, n_blocks, n_shots))

    return dev_name, n_blocks, n_shots


def cross_entropy(blocks: int, t: float):
    simulator = qml.device("default.qubit", wires=(3 * blocks + 2))

    @qml.qnode(simulator)
    def get_reduced_dm(block: int):
        return fqhe_circuit(n_blocks=blocks, obs=red_ij([block - 1, block, block + 1]), phi_i=phi(t=t, n_blocks=blocks))

    chi_max = None
    for block_ in tqdm(range(1, 1 + blocks)):
        rho_block = get_reduced_dm(block=block_)

        block_rank = np.linalg.matrix_rank(rho_block)
        if chi_max is None:
            chi_max = block_rank
        else:
            if block_rank > chi_max:
                chi_max = block_rank
            else:
                pass

    return np.log2(chi_max)


def cross_entropy_analysis(max_blocks, t_range):
    fig, ax = plt.subplots()
    ax.set_xlabel("blocks", rotation=0)
    ax.set_ylabel(r'$E_{\chi}$')
    ax.set_title(r'$E_{\chi} \textit{ as function of the number of blocks we try to simulate (using only 1)}$')

    for tt in t_range:
        block_vals = list(range(1, max_blocks + 1))
        cross_entropy_vals = [cross_entropy(blocks=num_blocks, t=tt) for num_blocks in range(1, max_blocks + 1)]
        ax.plot(block_vals, cross_entropy_vals, marker='o', label=r't={}'.format(tt))
        ax.legend()

    plt.show()

    return



if __name__ == '__main__':
    # cross_entropy_analysis(max_blocks=5, t_range=np.linspace(0, 1.2, 10)[1:])

    n_blocks = 7
    t = 0.5

    """
    Example: 
        In command line write "python fqhe.py -blocks 4 --shots 5
    """
    dev_name, n_blocks, n_shots = get_circuit()
    n_wires = 3 * n_blocks + 2

    try:
        run_option = {"local": local_device, "aws": braket_device}
        dev = run_option[dev_name](n_wires, n_shots)
    except KeyError or IndexError:
        dev = local_device(n_wires, n_shots)

    fqhe = qml.QNode(fqhe_circuit, dev)

    verify_ni(n_blocks, fqhe, n_shots=n_shots)
    # verify_nij(n_blocks, fqhe, n_shots)
