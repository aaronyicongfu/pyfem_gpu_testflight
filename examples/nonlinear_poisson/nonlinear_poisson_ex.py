import numpy as np
import sys
import matplotlib.pylab as plt
import pyfem


def nonlinear_poisson():
    
    # create a structure
    creator = pyfem.ProblemCreator(nnodes_x=32, nnodes_y=32)
    nodes, conn, X, dof_fixed = creator.create_poisson_problem()
    x = np.ones(10) / 10

    """ Compute u """
    quadrature = pyfem.QuadratureBilinear2D()
    basis = pyfem.BasisBilinear2D(quadrature)
    model = pyfem.NonlinearPoisson2D(
        nodes, X, conn, dof_fixed, None, quadrature, basis
    )
    assembler = pyfem.Assembler(model)

    # Plot the u and the v displacements
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    method = ["direct", "gmres"]
    for i in range(len(method)):
        u = assembler.solve_nonlinear(method=method[i], xdv=x)
        assembler.plot(u, ax[i], levels=20)
        ax[i].set_title(method[i] + " method-u")
    plt.show()

    return


if __name__ == "__main__":
    nonlinear_poisson()
