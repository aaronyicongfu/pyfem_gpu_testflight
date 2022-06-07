import sys

sys.path.append("../..")
import pyfem, utils
import numpy as np
from paropt import ParOpt
from mpi4py import MPI
import matplotlib.pylab as plt
import matplotlib.tri as tri
import argparse
from os import mkdir
from os.path import join, isdir


class TopoProblem(ParOpt.Problem):
    def __init__(
        self,
        model: (pyfem.LinearElasticity or pyfem.LinearPoisson),
        filtr: pyfem.Helmholtz,
        fixed_volume,
        save_history=False,
        save_history_every=10,
        prefix=".",
    ):
        ncon = 1
        super().__init__(MPI.COMM_SELF, model.nnodes, ncon)

        self.model = model
        self.filtr = filtr
        self.fixed_volume = fixed_volume
        self.save_history = save_history
        self.save_history_every = save_history_every
        self.prefix = prefix

        X = self.model.X
        lx = X[:, 0].max() - X[:, 0].min()
        ly = X[:, 1].max() - X[:, 1].min()
        self.fig, self.ax = plt.subplots(
            figsize=(4.8 * lx / ly, 4.8), constrained_layout=True
        )
        self.counter = 0
        return

    def getVarsAndBounds(self, x, lb, ub):
        x[:] = 0.95
        lb[:] = 1e-3
        ub[:] = 1.0
        return

    def evalObjCon(self, x):
        # x -> rho
        rho = self.filtr.apply(x)

        # Save the design
        if self.save_history and self.counter % self.save_history_every == 0:
            self.plot(rho, self.ax)
            self.fig.savefig(join(self.prefix, f"design_{self.counter:d}.pdf"))
        self.counter += 1

        # Evaluate compliance
        obj, self.u = self.model.compliance(rho)

        # Evaluate constraint
        con = [self.fixed_volume - self.model.volume(rho)]

        fail = 0
        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        # x -> rho
        rho = self.filtr.apply(x)
        g[:] = self.filtr.apply_gradient(self.model.compliance_grad(rho, self.u))
        A[0][:] = -self.filtr.apply_gradient(self.model.volume_grad(rho))
        fail = 0
        return fail

    def plot(self, u, ax, **kwargs):
        """
        Create a plot
        """
        # Clear old figure
        ax.clear()
        ax.axis("off")

        nnodes_per_elem = self.model.nnodes_per_elem
        nelems = self.model.nelems
        X = self.model.X
        # Create the triangles
        if nnodes_per_elem == 4:
            triangles = np.zeros((2 * nelems, 3), dtype=int)
            triangles[:nelems, 0] = conn[:, 0]
            triangles[:nelems, 1] = conn[:, 1]
            triangles[:nelems, 2] = conn[:, 2]

            triangles[nelems:, 0] = conn[:, 0]
            triangles[nelems:, 1] = conn[:, 2]
            triangles[nelems:, 2] = conn[:, 3]
        elif nnodes_per_elem == 3:
            triangles = conn
        else:
            raise ValueError("unsupported element type")

        # Create the triangulation object
        tri_obj = tri.Triangulation(X[:, 0], X[:, 1], triangles)

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, u, cmap="bwr", alpha=0.8, **kwargs)
        return


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument(
        "--problem", type=str, choices=["elasticity", "thermal"], default="elasticity"
    )
    p.add_argument(
        "--element_type", type=str, choices=["quad", "tri", "block"], default="quad"
    )
    p.add_argument("--r0", type=float, default=0.01, help="Filter radius")
    p.add_argument("--prefix", type=str, default="results")
    p.add_argument("--timer_threshold", type=float, default=10.0)
    args = p.parse_args()

    if not isdir(args.prefix):
        mkdir(args.prefix)

    save_history = True
    if args.element_type == "block":
        save_history = False

    utils.timer_set_threshold(
        args.timer_threshold
    )  # Don't print when t < threshold (unit: ms)

    # Create problem mesh
    if args.element_type == "quad":
        creator = pyfem.ProblemCreator(nnodes_x=128, nnodes_y=64, element_type="quad")
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)

    elif args.element_type == "tri":
        creator = pyfem.ProblemCreator(nnodes_x=128, nnodes_y=64, element_type="tri")
        quadrature = pyfem.QuadratureTriangle2D()
        basis = pyfem.BasisTriangle2D(quadrature)

    elif args.element_type == "block":
        creator = pyfem.ProblemCreator(
            nnodes_x=64, nnodes_y=32, nnodes_z=32, element_type="block"
        )
        quadrature = pyfem.QuadratureBlock3D()
        basis = pyfem.BasisBlock3D(quadrature)

    if args.problem == "elasticity":
        conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()
        model = pyfem.LinearElasticity(
            X, conn, dof_fixed, None, nodal_force, quadrature, basis, p=5.0
        )
    elif args.problem == "thermal":
        conn, X, dof_fixed = creator.create_poisson_problem()
        model = pyfem.LinearPoisson(
            X, conn, dof_fixed, None, quadrature, basis, gfunc=lambda x: 1.0, p=5.0
        )

    # Create the Helmholtz filter
    filtr = pyfem.Helmholtz(args.r0, X, conn, quadrature, basis)

    # Create paropt problem
    prob = TopoProblem(
        model, filtr, fixed_volume=0.4, save_history=save_history, prefix=args.prefix
    )

    options = {
        "algorithm": "mma",
        "mma_max_iterations": 200,
        "output_file": join(args.prefix, "paropt.out"),
        "tr_output_file": join(args.prefix, "paropt.tr"),
        "mma_output_file": join(args.prefix, "paropt.mma"),
    }

    # Set up the optimizer
    opt = ParOpt.Optimizer(prob, options)

    prob.checkGradients()
    opt.optimize()
    x, z, zw, zl, zu = opt.getOptimizedPoint()
    rho = filtr.apply(x)

    # Plot optimal design
    prob.plot(x, prob.ax)
    prob.fig.savefig(join(args.prefix, "opt_design_x.pdf"))
    prob.plot(rho, prob.ax)
    prob.fig.savefig(join(args.prefix, "opt_design_rho.pdf"))

    # Save design to vtk
    nodal_vals = {"x": x, "rho": rho}
    if args.problem == "thermal":
        c, u = model.compliance(rho)
        nodal_vals["T"] = u
    utils.to_vtk(conn, X, nodal_vals, vtk_name=join(args.prefix, "result.vtk"))
