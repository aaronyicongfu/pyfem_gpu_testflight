"""
This script shows how to perform a static, 3-dimensional linear elasticity
analysis on a structured mesh created by the built-in utility, and export a vtk
file for visualization.
"""
import sys

sys.path.append("../..")
import pyfem
import utils

# Set up the meshing utility and create the problem mesh
creator = pyfem.ProblemCreator(32, 8, 8, element_type="block")
nodes, conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()

# Set up 3-dimensional block quadrature and basis
quadrature = pyfem.QuadratureBlock3D()
basis = pyfem.BasisBlock3D(quadrature)

# Set up linear elasticity model
model = pyfem.LinearElasticity(
    nodes, X, conn, dof_fixed, None, nodal_force, quadrature, basis
)

# Set up assembler
assembler = pyfem.Assembler(model)

# Solve the linear system and extract directional nodal displacements
u = assembler.solve(method="direct")

ux = u[0::3]
uy = u[1::3]
uz = u[2::3]
sol = {"ux": ux, "uy": uy, "uz": uz}

utils.to_vtk(nodes, conn, X, sol)

print("summary")
print(f"ndof: {len(u)}")
