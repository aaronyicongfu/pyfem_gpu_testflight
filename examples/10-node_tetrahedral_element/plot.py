
"""
This script shows how to perform a static, 3-dimensional linear elasticity
analysis on a structured mesh created by the built-in utility, and export a vtk
file for visualization.
"""
import sys
import argparse

sys.path.append("../..")
import pyfem
import utils
from icecream import ic

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=35)
p.add_argument("--ny", type=int, default=9)
p.add_argument("--nz", type=int, default=9)
p.add_argument(
    "--assemble_only", action="store_true", help="assemble the matrix and exit"
)
p.add_argument("--threshold", type=float, default=100.0)
p.add_argument("--solver", default="cg", choices=["direct", "cg", "gmres"])
args = p.parse_args()

utils.timer_on()
utils.timer_set_threshold(args.threshold)

# Set up the meshing utility and create the problem mesh
creator = pyfem.ProblemCreator(args.nx, args.ny, args.nz, element_type="tet")
conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()
ic(conn)
# ic(conn.values())

# Set up 3-dimensional block quadrature and basis
quadrature = pyfem.QuadratureTetrahedron8Point()
basis = pyfem.BasisTetrahedron10node(quadrature)

# Set up linear elasticity model
model = pyfem.LinearElasticity(X, conn, dof_fixed, None, nodal_force, quadrature, basis, p=5.0)

if args.assemble_only:
    K = model.compute_jacobian()
    rhs = model.compute_rhs()
    K, rhs = model.apply_dirichlet_bcs(K, rhs, enforce_symmetric_K=True)

else:
    # Set up assembler
    assembler = pyfem.Assembler(model)

    # Solve the linear system and extract directional nodal displacements
    u = assembler.solve(method=args.solver)

    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    sol = {"ux": ux, "uy": uy, "uz": uz}

    
    utils.to_vtk(conn, X, sol)

print("summary")
print(f"nnodes: {X.shape[0]}")
print(f"ndof:   {X.shape[0]*X.shape[1]}")