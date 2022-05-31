import unittest
import sys
from ref_helmholtz import NodeFilter
from scipy.sparse.linalg import spsolve
import numpy as np

sys.path.append("..")
import pyfem


class Helmholtz2D(unittest.TestCase):
    def test_helmholtz_filter(self):
        # Set up problem
        creator = pyfem.ProblemCreator(nnodes_x=32, nnodes_y=32, element_type="quad")
        conn, X, x = creator.create_helmhotz_problem()
        r0 = 0.1

        np.random.seed(0)
        gradrho = np.random.rand(x.shape[0])

        # Compute u
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        model = pyfem.Helmholtz(0.1, X, conn, quadrature, basis)
        u = model.apply(x)
        gradx = model.apply_gradient(gradrho)

        # Compute u_ref
        filtr = NodeFilter(r0, conn, X)
        u_ref = filtr.apply(x)
        gradx_ref = filtr.applyGradient(gradrho)

        # Compare
        np.random.seed(123)
        p = np.random.rand(u.shape[0])
        pTu = p.dot(u)
        pTu_ref = p.dot(u_ref)
        pTg = p.dot(gradx)
        pTg_ref = p.dot(gradx_ref)
        print(f"pTu    :{pTu}")
        print(f"pTu_ref:{pTu_ref}")
        print(f"pTg    :{pTg}")
        print(f"pTg_ref:{pTg_ref}")
        self.assertAlmostEqual((pTu - pTu_ref) / pTu, 0, delta=1e-10)
        self.assertAlmostEqual((pTg - pTg_ref) / pTu, 0, delta=1e-10)

        return


if __name__ == "__main__":
    unittest.main()
