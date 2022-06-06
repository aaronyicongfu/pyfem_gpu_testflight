import numpy as np
import unittest
import sys
from ref_linear_poisson import Poisson
from ref_linear_poisson import gfunc as gfunc_ref

sys.path.append("..")
import pyfem


def gfunc(x):
    _x = x[..., 0]
    _y = x[..., 1]
    return _x * (_x - 5.0) * (_x - 10.0) * _y * (_y - 4.0)


class LinearPoissonCase(unittest.TestCase):
    def test_linear_poisson(self):
        # Compute u
        creator = pyfem.ProblemCreator(nnodes_x=32, nnodes_y=32)
        conn, X, dof_fixed = creator.create_poisson_problem()
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        model = pyfem.LinearPoisson(X, conn, dof_fixed, None, quadrature, basis, gfunc)
        assembler = pyfem.Assembler(model)
        u = assembler.solve(method="direct")

        # Compute u_ref
        poisson = Poisson(conn, X, dof_fixed, gfunc_ref)
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


class ElasticityDerivative(unittest.TestCase):
    def setUp(self):
        # Create creators
        self.creator_2d_quad = pyfem.ProblemCreator(
            nnodes_x=64, nnodes_y=64, element_type="quad"
        )
        self.creator_2d_tri = pyfem.ProblemCreator(
            nnodes_x=64, nnodes_y=64, element_type="tri"
        )
        self.creator_3d_block = pyfem.ProblemCreator(
            nnodes_x=8, nnodes_y=8, nnodes_z=8, element_type="block"
        )
        return

    def run_dKdx(self, creator, quadrature, basis):
        """
        Test the derivative of phi^T K psi w.r.t. nodal variable x
        """

        conn, X, dof_fixed = creator.create_poisson_problem()
        model = pyfem.LinearPoisson(
            X, conn, dof_fixed, None, quadrature, basis, gfunc, p=5.0
        )

        np.random.seed(0)
        nnodes = X.shape[0]
        ndof = X.shape[0]
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

    def run_compliance_gradient(self, creator, quadrature, basis):
        conn, X, dof_fixed = creator.create_poisson_problem()
        model = pyfem.LinearPoisson(
            X, conn, dof_fixed, None, quadrature, basis, gfunc, p=5.0
        )

        np.random.seed(0)
        nnodes = X.shape[0]
        rho = np.random.rand(nnodes)
        p = np.random.rand(nnodes)
        h = 1e-30

        c, u = model.compliance(rho, solver="cg")
        grad = model.compliance_grad(rho, u)
        grad = p.dot(grad)

        c_cs, _ = model.compliance(rho + 1j * p * h, solver="direct")
        grad_cs = c_cs.imag / h
        print("compliance:", c)
        print(f"grad:    {grad:.15e}")
        print(f"grad_cs: {grad_cs:.15e}")
        self.assertAlmostEqual((grad - grad_cs) / grad, 0.0, delta=1e-10)
        return

    def test_2d_quad(self):
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        self.run_dKdx(self.creator_2d_quad, quadrature, basis)
        self.run_compliance_gradient(self.creator_2d_quad, quadrature, basis)
        return

    def test_2d_tri(self):
        quadrature = pyfem.QuadratureTriangle2D()
        basis = pyfem.BasisTriangle2D(quadrature)
        self.run_dKdx(self.creator_2d_tri, quadrature, basis)
        self.run_compliance_gradient(self.creator_2d_tri, quadrature, basis)
        return

    def test_3d_block(self):
        quadrature = pyfem.QuadratureBlock3D()
        basis = pyfem.BasisBlock3D(quadrature)
        self.run_dKdx(self.creator_3d_block, quadrature, basis)
        self.run_compliance_gradient(self.creator_3d_block, quadrature, basis)
        return


if __name__ == "__main__":
    unittest.main()
