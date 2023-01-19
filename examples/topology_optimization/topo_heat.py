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
        model: pyfem.LinearPoisson,
        filtr: pyfem.Helmholtz,
        fixed_volume,
        save_history=False,
        save_history_every=10,
        prefix=".",
        weighted=True,
    ):
        ncon = 1
        super().__init__(MPI.COMM_SELF, model.nnodes, ncon)

        self.model = model
        self.filtr = filtr
        self.fixed_volume = fixed_volume
        self.save_history = save_history
        self.save_history_every = save_history_every
        self.prefix = prefix
        self.weighted = weighted

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
        obj, self.u = self.model.compliance(rho, weighted=self.weighted)

        # Evaluate constraint
        con = [self.fixed_volume - self.model.volume(rho)]

        fail = 0
        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        # x -> rho
        rho = self.filtr.apply(x)
        g[:] = self.filtr.apply_gradient(
            self.model.compliance_grad(rho, self.u, weighted=self.weighted)
        )
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
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--cold_wall", type=float, default=0.0)
    p.add_argument("--heat_source", type=float, default=1.0)
    p.add_argument("--no_weighted", action="store_false")
    p.add_argument("--check_gradient", action="store_true")
    p.add_argument("--r0", type=float, default=0.01, help="Filter radius")
    p.add_argument("--prefix", type=str, default="results")
    p.add_argument("--timer_threshold", type=float, default=10.0)
    args = p.parse_args()

    if not isdir(args.prefix):
        mkdir(args.prefix)

    save_history = True

    utils.timer_set_threshold(
        args.timer_threshold
    )  # Don't print when t < threshold (unit: ms)

    # Create problem mesh
    creator = pyfem.ProblemCreator(
        nnodes_x=args.nx, nnodes_y=args.ny, element_type="quad"
    )
    quadrature = pyfem.QuadratureBilinear2D()
    basis = pyfem.BasisBilinear2D(quadrature)

    dof_fixed = []
    # for j in range(3 * args.ny // 8, 5 * args.ny // 8):
    for j in range(args.ny):
        dof_fixed.append(creator.nodes3d[0, j, 0])  # fixed right edge
    conn, X = creator.conn, creator.X

    model = pyfem.LinearPoisson(
        X=X,
        conn=conn,
        dof_fixed=dof_fixed,
        dof_fixed_vals=len(dof_fixed) * [args.cold_wall],
        quadrature=quadrature,
        basis=basis,
        gfunc=lambda x: args.heat_source,
        p=5.0,
    )

    # Create the Helmholtz filter
    filtr = pyfem.Helmholtz(args.r0, X, conn, quadrature, basis)

    # Create paropt problem
    prob = TopoProblem(
        model,
        filtr,
        fixed_volume=0.4,
        save_history=save_history,
        prefix=args.prefix,
        weighted=args.no_weighted,
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
    if args.check_gradient:
        exit()

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
    c, u = model.compliance(rho)
    nodal_vals["T"] = u
    utils.to_vtk(conn, X, nodal_vals, vtk_name=join(args.prefix, "result.vtk"))
