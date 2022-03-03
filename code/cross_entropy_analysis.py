import pennylane as qml
from pennylane import numpy as np
from fqhe_prep import fqhe_circuit, phi
from tqdm import tqdm
from matplotlib import pyplot as plt


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
