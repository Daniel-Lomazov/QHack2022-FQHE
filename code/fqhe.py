from fqhe_verification_1pt import verify_ni, verify_state
from fqhe_verification_2pt import verify_nij, verify_nij_2
from fqhe_verification_string import verify_string
from fqhe_prep import get_argparse, get_circuit

if __name__ == '__main__':
    # cross_entropy_analysis(max_blocks=5, t_range=np.linspace(0, 1.2, 10)[1:])
    """
    Example: 
        In command line write "python3 fqhe.py --device cirq --blocks 5 --shots 3000 --obs 1pt"
    """
    sims = {"1pt": verify_ni, "2pt": verify_nij_2, "string":verify_string, "state": verify_state}
    
    args = get_argparse()

    fqhe_circuit, n_blocks, n_shots, = get_circuit(args)

    for s in args.obs:
        print("runinng " , s," simulation")
        sims[s](n_blocks, fqhe_circuit, n_shots)

    # compare_ni(n_blocks, fqhe)

    # verify_nij_2(n_blocks, fqhe, n_shots)
    # verify_ni(n_blocks, fqhe, n_shots)

    # verify_string(n_blocks, fqhe, n_shots=n_shots)
    # measure = measure_ni_2(n_blocks)
    # phi_i = phi(0.5, n_blocks)
    # print(qml.draw(fqhe)(n_blocks, measure, phi_i))
    # fig, ax = qml.draw_mpl(fqhe, decimals=2)(n_blocks, measure, phi_i)
    # plt.show()
