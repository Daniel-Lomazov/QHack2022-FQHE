import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import strftime
from fqhe_prep import phi

plt.rcParams['text.usetex'] = True


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


def measure_ninj(n_blocks, i=2):
    def ret():
        return [qml.sample(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)) for j in range(i + 1, 3 * (n_blocks + 1))]

    return ret


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
    tvals, t_output = verify_nij_data(n_blocks, fqhe_circuit)

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


def verify_nij_data2(n_blocks, fqhe_circuit, comulant_form=True):
    n_t = 31
    tvals = np.round(np.linspace(3.0, 0, n_t), 3)

    # Calculation of <n_i>
    measure = measure_zi(n_blocks)
    samples_zi_diff_t_vals = [np.array(fqhe_circuit(n_blocks, measure, phi(t, n_blocks))) for t in tqdm(tvals)]
    z_exp_vals_for_diff_t_vals = [np.average(sample, axis=1) for sample in samples_zi_diff_t_vals]

    # Calulation of <n_in_j>
    measure = measure_ninj(n_blocks)
    samples_zizj_diff_t_vals = [np.array(fqhe_circuit(n_blocks, measure, phi(t, n_blocks))) for t in tqdm(tvals)]
    zz_exp_vals_for_diff_t_vals = [np.average(sample, axis=1) for sample in samples_zizj_diff_t_vals]

    n2_exp_vals_for_diff_t_vals = []
    for t_idx in range(len(zz_exp_vals_for_diff_t_vals)):
        zz = zz_exp_vals_for_diff_t_vals[t_idx]

        i, j = 2, 3
        n2_t_idx = []
        for zz_idx in range(len(zz)):
            z_ij = zz[zz_idx]
            z_i = z_exp_vals_for_diff_t_vals[t_idx][i]
            z_j = z_exp_vals_for_diff_t_vals[t_idx][j]

            n_ij = np.absolute((1 - z_i - z_j + z_ij) / 4)

            if comulant_form is True:
                # comulant form n_ij := plain n_ij - n_i * n_j
                n_ij = np.absolute((z_ij - z_i * z_j) / 4)
            else:
                # plain old n_ij
                n_ij = np.absolute((1 - z_i - z_j + z_ij) / 4)

            n2_t_idx.append(n_ij)
            j = j + 1
        n2_exp_vals_for_diff_t_vals.append(np.array(n2_t_idx))

    return tvals, n2_exp_vals_for_diff_t_vals


def verify_nij_2(n_blocks, fqhe_circuit, n_shots, comulant_form: bool = True):
    """
    Verifies eq. 15 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """
    tvals, n2_exp_vals_for_diff_t_vals = verify_nij_data2(n_blocks, fqhe_circuit)
    fig, ax = plt.subplots()

    # Set correct x-ticks
    xticks_labels = range(1, 3 * (n_blocks + 1))
    xticks_vals = range(xticks_labels[-1] - xticks_labels[0] + 1)
    ax.set_xticks(np.array(list(xticks_vals)), labels=xticks_labels)

    # Set correct y-ticks
    # yticks_labels = tvals
    ax.set_yticks(np.arange(len(tvals)), labels=tvals)

    ax.set_xlabel('j - i (i=2)')
    ax.set_ylabel("t", rotation=0)
    ax.set_title('N = {}; shots = {} on {}'.format(3 * (n_blocks + 1), n_shots, fqhe_circuit.device.short_name))

    plt.grid(visible=True)
    im = ax.imshow(n2_exp_vals_for_diff_t_vals)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        r'$|\left \langle n_in_j \right \rangle - \left \langle n_i \right \rangle \left \langle n_j \right \rangle|$',
        rotation=90, va="bottom", labelpad=25)

    plt.savefig("2pt//2pt" + fqhe_circuit.device.short_name + "_" + str(n_blocks) +
                "blocks_" + str(n_shots) + "shots_" + strftime("%d%m_%H_%M") + ".png", format="png")
    plt.show()

    return