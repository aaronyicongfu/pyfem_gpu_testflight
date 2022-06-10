import numpy as np
import unittest
import sys

sys.path.append("..")
import pyfem


class ElasticityDerivative(unittest.TestCase):
    def setUp(self):
        self.creator_3d_tetra = pyfem.ProblemCreator(
            nnodes_x=3, nnodes_y=3, nnodes_z=3, element_type="tet"
        )
        return

    def run_dKdx(self, creator, quadrature, basis):
        """
        Test the derivative of phi^T K psi w.r.t. nodal variable x
        """
        (
            conn,
            X,
            dof_fixed,
            nodal_force,
        ) = creator.create_linear_elasticity_problem()
        model = pyfem.LinearElasticity(
            X, conn, dof_fixed, None, nodal_force, quadrature, basis, p=5.0
        )

        np.random.seed(0)
        nnodes = X.shape[0]
        ndof = X.shape[0] * X.shape[1]
        phi = np.random.rand(ndof)
        psi = np.random.rand(ndof)

        rho = np.random.rand(nnodes)
        p = np.random.rand(nnodes)
        h = 1e-30

        # Compute K derivative
        dfdrho = model._compute_K_dv_sens(rho, phi, psi)
        dfdrho = p.dot(dfdrho)

        # Compute K derivative via complex step
        K = model.compute_jacobian(rho + 1j * p * h)
        dfdrho_cs = phi.dot(K.dot(psi)).imag / h

        # Verify
        print(f"dfdrho   :{dfdrho:.16e}")
        print(f"dfdrho_cs:{dfdrho_cs:.16e}")
        self.assertAlmostEqual((dfdrho - dfdrho_cs) / dfdrho, 0.0, delta=1e-12)
        return

    def test_3d_block_tetrahedron(self):
        quadrature = pyfem.QuadratureTetrahedron8Point()
        basis = pyfem.BasisTetrahedron10node(quadrature)
        self.run_dKdx(self.creator_3d_tetra, quadrature, basis)
        return


if __name__ == "__main__":
    unittest.main()
