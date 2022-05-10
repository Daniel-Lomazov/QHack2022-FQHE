import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import strftime
from fqhe_prep import phi

plt.rcParams['text.usetex'] = True

def verify_state(n_blocks, fqhe_circuit, shots, t_bottom=0, t_top=1.2, n_t=11):
    fig, ax = qml.draw_mpl(fqhe_circuit)(n_blocks,qml.state, phi(0.5, n_blocks))
    plt.savefig("circuit.png")
    tvals = np.round(np.linspace(t_top, t_bottom, n_t), 3)
    
    with np.printoptions(threshold=2000, edgeitems=2000, linewidth=2000):
        for t in tqdm(tvals):
            state = np.array(fqhe_circuit(n_blocks, qml.state, phi(t, n_blocks)))
            state_not_0 = np.where(state != 0)
            print(state_not_0)
            print(state[state_not_0])


def verify_ni_data(n_blocks, fqhe_circuit, t_bottom=0, t_top=1.2, n_t=11):
    tvals = np.round(np.linspace(t_top, t_bottom, n_t), 3)
    measure = measure_zi(n_blocks)
    
    samples_for_diff_t_vals = []
    for t in tqdm(tvals):
        temp = np.array(fqhe_circuit(n_blocks, measure, phi(t, n_blocks)))
        # print([row.sum() for row in temp ])
        temp = np.array([row for row in temp if row.sum() == len(row)/3])
        samples_for_diff_t_vals.append(temp)
        
    samples_for_diff_t_vals = np.array(samples_for_diff_t_vals)
        
    # samples_for_diff_t_vals = np.array([fqhe_circuit(n_blocks, measure, phi(t, n_blocks)) for t in tqdm(tvals)])
    print(samples_for_diff_t_vals.shape)

    z_exp_vals_for_diff_t_vals = np.average(samples_for_diff_t_vals, axis=2)
    print(np.sum(samples_for_diff_t_vals, axis=1))
    print(z_exp_vals_for_diff_t_vals.shape)
    n1_exp_vals_for_diff_t_vals = (1 - np.array(z_exp_vals_for_diff_t_vals)) / 2

    return tvals, n1_exp_vals_for_diff_t_vals


def verify_ni(n_blocks, fqhe_circuit, n_shots):
    """
    Verifies eqs. 15 & 16 from Rahmani et al.

    Parameters
    ----------
    n_blocks
    n_shots

    Returns
    -------

    """
    tvals, n1_exp_vals_for_diff_t_vals = verify_ni_data(n_blocks, fqhe_circuit)

    fig, ax = plt.subplots()
    im = ax.imshow(n1_exp_vals_for_diff_t_vals)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries

    ax.set_yticks(np.arange(len(tvals)))
    ax.set_yticklabels(tvals)
    ax.set_ylabel("t", rotation=0)
    ax.set_xticks(np.arange(3 * (n_blocks + 1)))
    ax.set_xticklabels(range(3 * (n_blocks + 1)))
    ax.set_xlabel(r'$\textit{\large{k}}$')
    ax.set_title('N = {}; shots = {} on {}'.format(3 * (n_blocks + 1), n_shots, fqhe_circuit.device.short_name))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r'$\left\langle n_k \right\rangle$', rotation=0, va="bottom")
    plt.savefig("1pt//1pt_" + fqhe_circuit.device.short_name + "_" + str(n_blocks) +
                "blocks_" + str(n_shots) + "shots_" + strftime("%d%m_%H_%M") + ".png", format="png")
    plt.show()

    return


def measure_zi(n_blocks, measurement=qml.sample):
    def ret():
        return [measurement(qml.PauliZ(wires=i)) for i in range(3 * (n_blocks + 1))]

    return ret