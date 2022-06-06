import numpy as np
import unittest
import sys
from ref_nonlinear_poisson import PoissonProblem, NonlinearPoisson
import matplotlib.pylab as plt

sys.path.append("..")
import pyfem


class NonLinearPoissonCase(unittest.TestCase):
    def test_nonlinear_poisson(self):

        # create a structure
        creator = pyfem.ProblemCreator(nnodes_x=32, nnodes_y=32)
        conn, X, dof_fixed = creator.create_poisson_problem()

        """ Compute u_ref """
        problem = PoissonProblem(10)
        # Create the Poisson problem
        poisson = NonlinearPoisson(conn, X, dof_fixed, problem)
        x = np.ones(problem.N) / problem.N
        u_ref = poisson.solve(x)

        """ Compute u """
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        model = pyfem.NonlinearPoisson2D(X, conn, dof_fixed, None, quadrature, basis)
        assembler = pyfem.Assembler(model)

        method = ["direct", "gmres"]
        for i in range(len(method)):
            u = assembler.solve_nonlinear(method=method[i], xdv=x)

            """ Comparsion """
            np.random.seed(123)
            p = np.random.rand(u.shape[0])
            pTu = p.dot(u)
            pTu_ref = p.dot(u_ref)
            print(f"pTu    :{pTu}")
            print(f"pTu_ref:{pTu_ref}")
            self.assertAlmostEqual((pTu - pTu_ref) / pTu, 0, delta=1e-8)

            # # Plot the u and the v displacements
            # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            # poisson.plot(u_ref, ax[0], levels=20)
            # ax[0].set_title(method[i] + "-u_ref")
            # assembler.plot(u, ax[1], levels=20)
            # ax[1].set_title(method[i] + "-u")
            # plt.show()

        return


if __name__ == "__main__":
    unittest.main()
