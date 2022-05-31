import sys

sys.path.append("../..")
import pyfem, utils
import numpy as np
from paropt import ParOpt
from mpi4py import MPI
import matplotlib.pylab as plt
import matplotlib.tri as tri


class TopoProblem(ParOpt.Problem):
    def __init__(
        self,
        model: pyfem.LinearElasticity,
        filtr: pyfem.Helmholtz,
        fixed_volume,
        save_history=True,
        save_history_every=10,
    ):
        ncon = 1
        super().__init__(MPI.COMM_SELF, model.nnodes, ncon)

        self.model = model
        self.filtr = filtr
        self.fixed_volume = fixed_volume
        self.save_history = save_history
        self.save_history_every = save_history_every

        self.fig, self.ax = plt.subplots(constrained_layout=True)
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
            self.fig.savefig(f"design_{self.counter:d}.png")
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
        ax.tricontourf(tri_obj, u, **kwargs)
        return


if __name__ == "__main__":
    utils.timer_set_threshold(10.0)  # Don't print t < 10 ms

    # Create linear elasticity model
    creator = pyfem.ProblemCreator(nnodes_x=128, nnodes_y=128, element_type="quad")
    conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()
    quadrature = pyfem.QuadratureBilinear2D()
    basis = pyfem.BasisBilinear2D(quadrature)
    # quadrature = pyfem.QuadratureTriangle2D()
    # basis = pyfem.BasisTriangle2D(quadrature)
    # quadrature = pyfem.QuadratureBlock3D()
    # basis = pyfem.BasisBlock3D(quadrature)

    model = pyfem.LinearElasticity(
        X, conn, dof_fixed, None, nodal_force, quadrature, basis, p=5.0
    )

    # Create the Helmholtz filter
    r0 = 0.01
    filtr = pyfem.Helmholtz(r0, X, conn, quadrature, basis)

    # Create paropt problem
    prob = TopoProblem(model, filtr, fixed_volume=0.4)

    options = {"algorithm": "mma", "mma_max_iterations": 100}

    # Set up the optimizer
    opt = ParOpt.Optimizer(prob, options)

    # Set a new starting point
    opt.optimize()
    x, z, zw, zl, zu = opt.getOptimizedPoint()
    rho = filtr.apply(x)
    utils.to_vtk(conn, X, {"x": x, "rho": rho})
