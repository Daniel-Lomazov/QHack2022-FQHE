from fqhe_verification import *
import argparse
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

plt.rcParams['text.usetex'] = True


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
    A measurement of v=1/3 FQHE state according to the observables specified by obs

    """
    if phi_i is None:
        raise ValueError("Must provide phi")

    # Stage 0
    for i in range(n_blocks + 1):
        qml.PauliX(wires=(3 * i))

    qml.Barrier(wires=range(3 * (n_blocks + 1)))

    # Stage 1
    qml.RY(-2 * phi_i[0], wires=[1])
    for i in range(1, n_blocks):
        qml.CRY(-2 * phi_i[i], wires=[3 * (i - 1) + 1, 3 * i + 1])

    qml.Barrier(wires=range(3 * (n_blocks + 1)))

    # Stage 2 - part 1
    for i in range(n_blocks):
        qml.CNOT(wires=[3 * i + 1, 3 * i + 2])

    # Stage 2 - part 2
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=(3 * i + 1))
        qml.CNOT(wires=[3 * i + 2, 3 * i + 3])

    # Stage 2 - part 3
    for i in range(n_blocks):
        qml.CNOT(wires=[3 * i + 1, 3 * i])

    qml.Barrier(wires=range(3 * (n_blocks + 1)))

    return obs()
    # return qml.expval(qml.PauliZ(0))


def local_device(n_wires, n_shots, name=""):
    dev = qml.device("default.qubit", wires=n_wires, shots=n_shots)
    return dev


def braket_device(n_wires, n_shots, name):
    my_bucket = "amazon-braket-amazon-braket-47ba137bf31d"
    my_prefix = "penny"
    s3_folder = (my_bucket, my_prefix)

    sv1 = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    aspn11 = "arn:aws:braket:::device/qpu/rigetti/Aspen-11"
    aspenm1 = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-1"

    dev_arn = {"sv1": sv1, 'aspen11': aspn11, "aspenm1": aspenm1}

    try:
        dev_remote = qml.device('braket.aws.qubit', device_arn=dev_arn[name], wires=n_wires, shots=n_shots)
    except KeyError or IndexError:
        dev_remote = qml.device('braket.aws.qubit', device_arn=sv1, wires=n_wires, shots=n_shots)

    print(dev_remote._device.name)
    return dev_remote


def get_circuit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="local")
    parser.add_argument('--name', type=str, default="aspen11")
    parser.add_argument('--blocks', type=int, default=7)
    parser.add_argument('--shots', type=int, default=1000)
    args = parser.parse_args()
    n_blocks = args.blocks
    n_shots = args.shots
    dev_name = args.device
    name = args.name

    print("Running FQHE v=1/3 simulation with device-{} on {} blocks for {} shots".format(dev_name + "-" + args.name,
                                                                                          n_blocks, n_shots))

    return dev_name, n_blocks, n_shots, name


def cross_entropy(blocks: int, t: float):
    simulator = qml.device("default.qubit", wires=(3 * blocks + 3))

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
    """
    Example: 
        In command line write "python fqhe.py --blocks 4 --shots 5
    """
    dev_name, n_blocks, n_shots, name = get_circuit()
    n_wires = 3 * (n_blocks + 1)

    # try:
    #     print(dev_name)
    #     run_option = {"local": local_device, "aws": braket_device}
    #     dev = run_option[dev_name](n_wires, n_shots, name)
    #     print(dev)
    # except KeyError or IndexError:
    #     dev = local_device(n_wires, n_shots)

    dev = braket_device(n_wires, n_shots, name)
    fqhe = qml.QNode(fqhe_circuit, dev)
    # verify_string(n_blocks, fqhe, n_shots=n_shots)
    # measure = measure_ni_2(n_blocks)
    # phi_i = phi(0.5, n_blocks)
    # print(qml.draw(fqhe)(n_blocks, measure, phi_i))
    # fig, ax = qml.draw_mpl(fqhe, decimals=2)(n_blocks, measure, phi_i)
    # plt.show()

    verify_nij_2(n_blocks, fqhe, n_shots)
    # verify_ni(n_blocks, fqhe, n_shots)
