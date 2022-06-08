import numpy as np
import sys

sys.path.append("../../..")
import pyfem

sys.path.append("../../../../a2d/examples/elasticity")

import example as a2d
import matplotlib.pyplot as plt
import os


def get_diff(n):
    creator = pyfem.ProblemCreator(n, n, n, element_type="block")
    conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()

    """
    Linear/Nonlinear elasticity problem
    """

    # Compute element-wise Jacobian using a2d
    problem_info = {"type": "elasticity", "E": 10, "nu": 0.3}
    model_a2d = pyfem.A2DWrapper(X, conn, dof_fixed, None, a2d, problem_info)
    K_a2d = model_a2d.compute_jacobian()

    # Compute element-wise Jacobian using pyfem
    quadrature = pyfem.QuadratureBlock3D()
    basis = pyfem.BasisBlock3D(quadrature)
    model = pyfem.LinearElasticity(
        X, conn, dof_fixed, None, nodal_force, quadrature, basis
    )
    K = model.compute_jacobian()

    # Check difference
    dof = model.ndof_per_node * model.nnodes
    abserr = np.max(np.abs(K - K_a2d))
    relerr = abserr / np.max(np.abs(K))
    print(
        f"[{dof:8d}]rel: {relerr:.10e}, abs: {abserr:.10e}, max(K): {np.max(np.abs(K)):.10e}"
    )

    return dof, abserr, relerr


if __name__ == "__main__":
    dof = []
    abserr = []
    relerr = []
    for n in [4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80]:
        _dof, _abserr, _relerr = get_diff(n)
        dof.append(_dof)
        abserr.append(_abserr)
        relerr.append(_relerr)

    mpl_style_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "aiaa.mplstyle"
    )
    plt.style.use(mpl_style_path)

    textwidth = 6.5  # in
    figwidth = textwidth / 2
    figheight = figwidth * 0.75
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(figwidth, figheight),
        constrained_layout=True,
    )

    ax.loglog(
        dof,
        abserr,
        color="blue",
        marker="o",
        markerfacecolor="white",
        linewidth=1.5,
        markeredgewidth=1.5,
        markersize=5.0,
        linestyle="None",
        label="absolute",
    )

    ax.loglog(
        dof,
        relerr,
        color="red",
        marker="o",
        markerfacecolor="white",
        linewidth=1.5,
        markeredgewidth=1.5,
        markersize=5.0,
        linestyle="None",
        label="relative",
    )

    ax.legend()
    ax.grid(which="major")
    ax.set_xlabel("Problem size")
    ax.set_ylabel("Error")
    fig.savefig("elasticity_precision.pdf")
