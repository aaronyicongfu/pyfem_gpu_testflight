import numpy as np
import unittest
import sys

sys.path.append("..")
import pyfem


def ref_plane_stress(conn, X, dof_fixed, nodal_force):
    from ref_plane_stress import PlaneStress

    bcs = {}
    for dof_idx in dof_fixed:
        node_idx = dof_idx // 2
        if node_idx not in bcs.keys():
            bcs[node_idx] = [0, 1]
    ps = PlaneStress(conn, X, bcs, nodal_force)
    u_ref = ps.solve()
    return u_ref


class PlaneStress(unittest.TestCase):
    def test_plane_stress(self):
        # Compute u
        creator = pyfem.ProblemCreator(nnodes_x=32, nnodes_y=32)
        (
            conn,
            X,
            dof_fixed,
            nodal_force,
        ) = creator.create_linear_elasticity_problem()
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        model = pyfem.LinearElasticity(
            X, conn, dof_fixed, None, nodal_force, quadrature, basis
        )
        assembler = pyfem.Assembler(model)
        u = assembler.solve(method="direct")

        # Compute u_ref
        u_ref = ref_plane_stress(conn, X, dof_fixed, nodal_force)

        # Compare
        np.random.seed(123)
        p = np.random.rand(u.shape[0])
        pTu = p.dot(u)
        pTu_ref = p.dot(u_ref)
        print(f"pTu    :{pTu}")
        print(f"pTu_ref:{pTu_ref}")
        self.assertAlmostEqual((pTu - pTu_ref) / pTu, 0.0, delta=1e-10)
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

    def run_case(self, creator, quadrature, basis):
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

    def test_2d_quad(self):
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        self.run_case(self.creator_2d_quad, quadrature, basis)
        return

    def test_2d_tri(self):
        quadrature = pyfem.QuadratureTriangle2D()
        basis = pyfem.BasisTriangle2D(quadrature)
        self.run_case(self.creator_2d_tri, quadrature, basis)
        return

    def test_3d_block(self):
        quadrature = pyfem.QuadratureBlock3D()
        basis = pyfem.BasisBlock3D(quadrature)
        self.run_case(self.creator_3d_block, quadrature, basis)
        return


if __name__ == "__main__":
    unittest.main()
