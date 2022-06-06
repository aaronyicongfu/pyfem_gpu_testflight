"""
This script shows how to perform a static, 2-dimensional plane stress analysis
using both quadrilateral and triangular element and visualize the results
"""
import sys

sys.path.append("../..")
import pyfem
import matplotlib.pyplot as plt
import utils


def run_problem(element_type):
    # Use the problem creator utility to create a 2-dimensional rectangular domain
    # with boundary condition and loads
    creator = pyfem.ProblemCreator(nnodes_x=96, nnodes_y=96, element_type=element_type)
    (
        conn,
        X,
        dof_fixed,
        nodal_force,
    ) = creator.create_linear_elasticity_problem()

    # Set quadrature and basis
    if element_type == "quad":
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
    else:
        quadrature = pyfem.QuadratureTriangle2D()
        basis = pyfem.BasisTriangle2D(quadrature)

    # Create the physical model
    model = pyfem.LinearElasticity(
        X, conn, dof_fixed, None, nodal_force, quadrature, basis
    )
    assembler = pyfem.Assembler(model)

    # Solve
    u = assembler.solve(method="direct")

    # Compute magnitude
    ux = u[0::2]  # Extract x-directional nodal displacement
    uy = u[1::2]  # Extract y-directional nodal displacement
    umag = (ux**2 + uy**2) ** 0.5

    # Save the mesh and solution to vtk
    utils.to_vtk(conn, X, {"ux": ux, "uy": uy}, vtk_name=f"{element_type}.vtk")
    return umag, assembler


if __name__ == "__main__":
    # Switch on timer
    utils.timer_on()

    # Plot
    fig, ax = plt.subplots(ncols=2, constrained_layout=True)
    uquad, assembler = run_problem("quad")
    utri, assembler = run_problem("tri")

    assembler.plot(uquad, ax[0], levels=50)
    assembler.plot(utri, ax[1], levels=50)
    ax[0].set_title(r"quadrilateral element")
    ax[1].set_title(r"triangular element")
    plt.show()
