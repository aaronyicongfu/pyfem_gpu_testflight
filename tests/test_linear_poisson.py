import numpy as np
import unittest
import sys
from ref_linear_poisson import Poisson, gfunc

sys.path.append("..")
import pyfem


class LinearPoissonCase(unittest.TestCase):
    def test_linear_poisson(self):
        # Compute u
        creator = pyfem.ProblemCreator(nnodes_x=32, nnodes_y=32)
        conn, X, dof_fixed = creator.create_poisson_problem()
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        model = pyfem.LinearPoisson2D(X, conn, dof_fixed, None, quadrature, basis)
        assembler = pyfem.Assembler(model)
        u = assembler.solve(method="direct")

        # Compute u_ref
        poisson = Poisson(conn, X, dof_fixed, gfunc)
        u_ref = poisson.solve()

        # Compare
        np.random.seed(123)
        p = np.random.rand(u.shape[0])
        pTu = p.dot(u)
        pTu_ref = p.dot(u_ref)
        print(f"pTu    :{pTu}")
        print(f"pTu_ref:{pTu_ref}")
        self.assertAlmostEqual((pTu - pTu_ref) / pTu, 0, delta=1e-10)
        return


if __name__ == "__main__":
    unittest.main()
