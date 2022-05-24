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


class PlaneStressCase(unittest.TestCase):
    def test_plane_stress(self):
        # Compute u
        creator = pyfem.ProblemCreator(nnodes_x=32, nnodes_y=32)
        (
            nodes,
            conn,
            X,
            dof_fixed,
            nodal_force,
        ) = creator.create_linear_elasticity_problem()
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        model = pyfem.LinearElasticity(
            nodes, X, conn, dof_fixed, None, nodal_force, quadrature, basis
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
        self.assertAlmostEqual((pTu - pTu_ref) / pTu, 0, delta=1e-10)
        return


if __name__ == "__main__":
    unittest.main()
