import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from fqhe import phi


def measure_string_ij(n_block, i, j):
    def ret():
        return None

    return None


def verify_string_data(n_blocks, fqhe_circuit):
    n_t = 13
    tvals = np.round(np.linspace(0, 1.2, n_t), 3)

    t_output = []

    for t in tvals:
        n0i = fqhe_circuit(n_blocks, measure_string_ij, phi(t, n_blocks))
        t_output.append(n0i)

    return tvals, t_output


def verify_string(n_blocks, fqhe_circuit, n_shots):
    """
    Verifies eq. 16 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """
    tvals, t_output = verify_string_data(n_blocks, fqhe_circuit, n_shots)

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

    n_t = 11
    tvals = np.round(np.linspace(0, 1.2, n_t), 3)
    typename = type(fqhe_circuit.device).__dict__['name']
    measure = measure_ni_2(n_blocks)
    samples_for_diff_t_vals = [np.array(fqhe_circuit(n_blocks, measure, phi(t, n_blocks))) for t in tqdm(tvals)]
    tvals = np.flip(tvals)

    z_exp_vals_for_diff_t_vals = []
    for sample in samples_for_diff_t_vals:
        z_i_vals = []
        for obs_idx in range(sample.shape[0]):
            z_i_data = sample[obs_idx]
            z_i = sum(z_i_data) / len(z_i_data)
            z_i_vals.append(z_i)
        z_exp_vals_for_diff_t_vals.append(np.array(z_i_vals))

    n1_exp_vals_for_diff_t_vals = [((1 - o) / 2) for o in z_exp_vals_for_diff_t_vals]
    n1_exp_vals_for_diff_t_vals = np.flip(np.stack(n1_exp_vals_for_diff_t_vals), axis=0)

    fig, ax = plt.subplots()
    im = ax.imshow(n1_exp_vals_for_diff_t_vals)
    plt.grid(visible=True)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(n_t), labels=tvals)

    ax.set_xticks(np.arange(3 * (n_blocks + 1)), labels=range(3 * (n_blocks + 1)))
    ax.set_xlabel(r'$\textit{\large{k}}$')
    ax.set_ylabel("t", rotation=0)
    ax.set_title('N = {}; shots = {}'.format(3 * (n_blocks + 1), n_shots))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r'$\left\langle n_k \right\rangle$', rotation=90, va="bottom")
    plt.show()

    return


def verify_nij_2(n_blocks, fqhe_circuit, n_shots, comulant_form: bool=True):
    """
    Verifies eq. 15 from Rahmani et al.

    Parameters
    ----------
    n_blocks

    Returns
    -------

    """
    n_t = 30
    tvals = np.round(np.linspace(0, 3.0, n_t), 1)

    typename = type(fqhe_circuit.device).__dict__['name']

    measure = measure_ni_2(n_blocks)
    samples_for_diff_t_vals = [np.array(fqhe_circuit(n_blocks, measure, phi(t, n_blocks))) for t in tqdm(tvals)]

    z_exp_vals_for_diff_t_vals = []
    for sample in samples_for_diff_t_vals:
        z_i_vals = []
        for obs_idx in range(sample.shape[0]):
            z_i_data = sample[obs_idx]
            z_i = sum(z_i_data) / len(z_i_data)
            z_i_vals.append(z_i)
        z_exp_vals_for_diff_t_vals.append(np.array(z_i_vals))

    measure = measure_nij_2(n_blocks)
    samples_for_diff_t_vals = [np.array(fqhe_circuit(n_blocks, measure, phi(t, n_blocks))) for t in tqdm(tvals)]

    zz_exp_vals_for_diff_t_vals = []
    for sample in samples_for_diff_t_vals:
        z_ij_vals = []
        for obs_idx in range(sample.shape[0]):
            z_ij_data = sample[obs_idx]
            z_ij = sum(z_ij_data) / len(z_ij_data)
            z_ij_vals.append(z_ij)
        zz_exp_vals_for_diff_t_vals.append(np.array(z_ij_vals))
    tvals = np.flip(tvals)

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

    n2_exp_vals_for_diff_t_vals = np.flip(np.stack(n2_exp_vals_for_diff_t_vals), axis=0)

    fig, ax = plt.subplots()

    # Set correct x-ticks
    xticks_labels = range(1, 3 * (n_blocks + 1))
    xticks_vals = range(xticks_labels[-1] - xticks_labels[0] + 1)
    ax.set_xticks(np.array(list(xticks_vals)), labels=xticks_labels)

    # Set correct y-ticks
    # yticks_labels = tvals
    ax.set_yticks(np.arange(n_t), labels=tvals)

    ax.set_xlabel('j - i (i=2)')
    ax.set_ylabel("t", rotation=0)
    ax.set_title('N = {}; shots = {}'.format(3 * (n_blocks + 1), n_shots))

    plt.grid(visible=True)
    im = ax.imshow(n2_exp_vals_for_diff_t_vals)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        r'$|\left \langle n_in_j \right \rangle - \left \langle n_i \right \rangle \left \langle n_j \right \rangle|$',
        rotation=90, va="bottom")
    plt.show()

    return

# def measure_ni(n_blocks, typename=""):
#     def ret():
#         if "Aws" in typename:
#             obs = [qml.PauliZ(wires=i) for i in range(3 * (n_blocks + 1))]
#         else:
#             obs = [qml.Hermitian(0.5 * (qml.Identity(wires=i).matrix - qml.PauliZ(wires=i).matrix), wires=i) for i in
#                    range(3 * (n_blocks + 1))]
#         return [qml.expval(o) for o in obs]
#
#     return ret


def measure_ni_2(n_blocks):
    def ret():
        return [qml.sample(qml.PauliZ(wires=i)) for i in range(3 * (n_blocks + 1))]
    return ret


def measure_nij_2(n_blocks, i=2):
    def ret():
        to_return = []
        for j in range(3 * (n_blocks + 1)):
            if i < j:
                to_return.append(qml.sample(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)))
            else:
                pass
        return to_return
    return ret
