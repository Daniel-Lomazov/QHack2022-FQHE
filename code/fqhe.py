from fqhe_verification import *
from fqhe_prep import *
import pennylane as qml
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True


if __name__ == '__main__':
    # cross_entropy_analysis(max_blocks=5, t_range=np.linspace(0, 1.2, 10)[1:])
    """
    Example: 
        In command line write "python fqhe.py --blocks 4 --shots 5
    """
    dev, n_blocks, n_shots, = get_circuit()

    fqhe = qml.QNode(fqhe_circuit, dev)

    # compare_ni(n_blocks, fqhe)


    # verify_nij_2(n_blocks, fqhe, n_shots)
    verify_ni(n_blocks, fqhe, n_shots)

    # verify_string(n_blocks, fqhe, n_shots=n_shots)
    # measure = measure_ni_2(n_blocks)
    # phi_i = phi(0.5, n_blocks)
    # print(qml.draw(fqhe)(n_blocks, measure, phi_i))
    # fig, ax = qml.draw_mpl(fqhe, decimals=2)(n_blocks, measure, phi_i)
    # plt.show()
