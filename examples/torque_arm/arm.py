"""
This script shows how to perform a static, 2-dimensional linear elasticity
analysis on a mesh from abaqus input file.
"""
import sys
from os.path import abspath, join, dirname

sys.path.append("../..")
import pyfem, utils
import matplotlib.pyplot as plt
import numpy as np
from parse_inp import InpParser

if __name__ == "__main__":
    # Switch on timer
    utils.timer_on()

    # Parse the inp file
    inp_file = "arm.inp"
    inp_parser = InpParser(join(dirname(abspath(__file__)), inp_file))
    conn, X, groups = inp_parser.parse()

    # Since the problem is 2-dimensional, we only need first 2 coordinates
    X = X[:, 0:2]

    # We only need to use the triangle element (CPS3 element type in Abaqus)
    conn = conn["CPS3"]

    # We fix the x and y-directional dof for node group "fixed"
    dof_fixed = np.concatenate((2 * groups["fixed"], 2 * groups["fixed"] + 1))

    # We apply downward force on node group "load"
    nodal_force = {n: [0.0, -1.0] for n in groups["load"]}

    # Set triangular quadrature and basis accordingly
    quadrature = pyfem.QuadratureTriangle2D()
    basis = pyfem.BasisTriangle2D(quadrature)

    # Create the 2-dimensional plane stress physical model
    model = pyfem.LinearElasticity(
        X, conn, dof_fixed, None, nodal_force, quadrature, basis
    )

    # Create the problem assembler
    assembler = pyfem.Assembler(model)

    # Solve for displacement field
    u = assembler.solve(method="direct")

    # Plot
    fig, ax = plt.subplots(nrows=2, constrained_layout=True)
    ux = u[0::2]  # Extract x-directional nodal displacement
    uy = u[1::2]  # Extract y-directional nodal displacement
    assembler.plot(ux, ax[0], levels=50)
    assembler.plot(uy, ax[1], levels=50)
    ax[0].set_title(r"$u_x$")
    ax[1].set_title(r"$u_y$")
    plt.show()

    # vtk can be generated as follows to visualize geometry and solution in ParaView
    inp_parser.to_vtk({"ux": ux, "uy": uy})
