from re import A
import sys
import argparse

from matplotlib import markers

sys.path.append("../../..")
import pyfem
import utils
from pprint import pprint

sys.path.append("../../../../a2d/examples/elasticity")
import example as a2d

import matplotlib.pyplot as plt
import os


def gfunc(x):
    _x = x[..., 0]
    _y = x[..., 1]
    return _x * (_x - 5.0) * (_x - 10.0) * _y * (_y - 4.0)


def run_assemble_case(nx, ny, nz, problem):
    creator = pyfem.ProblemCreator(
        nnodes_x=nx, nnodes_y=ny, nnodes_z=nz, element_type="block"
    )
    quadrature = pyfem.QuadratureBlock3D()
    basis = pyfem.BasisBlock3D(quadrature)

    if problem == "elasticity":
        conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()
        model = pyfem.LinearElasticity(
            X, conn, dof_fixed, None, nodal_force, quadrature, basis, E=10.0, nu=0.3
        )
        problem_info = {"type": "elasticity", "E": 10.0, "nu": 0.3}
        a2dmodel = pyfem.A2DWrapper(X, conn, dof_fixed, None, a2d, problem_info)
    else:
        conn, X, x = creator.create_helmhotz_problem()
        r0 = 0.05
        model = pyfem.Helmholtz(r0, X, conn, quadrature, basis)
        problem_info = {"type": "helmholtz", "r0": r0}
        a2dmodel = pyfem.A2DWrapper(X, conn, [], None, a2d, problem_info)

    # Assemble element jacobian with pyfem
    if problem == "elasticity":
        model._compute_element_jacobian(model.Ke_mat)

    # Assemble element jacobian with a2d
    a2dmodel.compute_jacobian()

    # Get problem size
    ndof = model.ndof_per_node * model.nnodes

    return ndof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--problem", default="helmholtz", choices=["elasticity", "helmholtz"]
    )
    args = p.parse_args()

    # Switch on timer
    utils.timer_on()
    utils.timer_set_threshold(0.0)

    nx = [32, 48, 64, 80]
    ny = [32, 48, 64, 80]
    nz = [32, 48, 64, 80]
    # nx = [2, 3, 4, 6]
    # ny = [2, 3, 4, 6]
    # nz = [2, 3, 4, 6]
    ndof = []
    for _nx, _ny, _nz in zip(nx, ny, nz):
        _ndof = run_assemble_case(_nx, _ny, _nz, args.problem)
        ndof.append(_ndof)

    pprint(utils.MyProfiler.saved_times)
    print(len(utils.MyProfiler.saved_times))

    mpl_style_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "aiaa.mplstyle"
    )
    plt.style.use(mpl_style_path)

    fig, ax = plt.subplots(
        ncols=1, nrows=1, figsize=(3.3, 2.5), constrained_layout=True
    )

    a2d_time = utils.MyProfiler.saved_times["A2DWrapper._compute_jacobian_tensor"]
    if args.problem == "elasticity":
        pyfem_time = utils.MyProfiler.saved_times[
            "LinearElasticity._einsum_element_jacobian"
        ]
    else:
        pyfem_time = utils.MyProfiler.saved_times["Helmholtz._einsum_element_jacobian"]

    ax.loglog(
        ndof,
        pyfem_time,
        color="blue",
        marker="o",
        markerfacecolor="white",
        linewidth=1.5,
        markeredgewidth=1.5,
        markersize=5.0,
        label="pyfem",
    )
    ax.loglog(
        ndof,
        a2d_time,
        color="red",
        marker="o",
        markerfacecolor="white",
        linewidth=1.5,
        markeredgewidth=1.5,
        markersize=5.0,
        label="A2D",
    )

    ax.legend()
    ax.set_xlabel("problem size (number of degrees of freedom)")
    ax.set_ylabel("Execution time (ms)")
    ax.grid(which="major")

    fig.savefig(f"Ke_time_{args.problem}.pdf")
