"""
Utility functions for profiling, data scattering, coordinate transformation, etc.
"""
from time import perf_counter_ns
import numpy as np


def time_this(func):
    """
    Decorator: time the execution of a function
    """
    tab = "    "
    time_this.counter = 0  # a "static" variable
    fun_name = func.__qualname__

    def wrapper(*args, **kwargs):
        info_str = f"{tab*time_this.counter}{fun_name}() called"
        print(f"[timer] {info_str:<40s}")
        time_this.counter += 1
        t0 = perf_counter_ns()
        ret = func(*args, **kwargs)
        t1 = perf_counter_ns()
        time_this.counter -= 1
        info_str = f"{tab*time_this.counter}{fun_name}() return"
        print(
            f"[timer] {info_str:<80s}",
            f"({(t1 - t0) / 1e6:.2f} ms)",
        )
        return ret

    return wrapper


@time_this
def scatter_node_to_elem(conn, data, data_e):
    """
    Scatter (scalar or vector) nodal quantities to elements: data -> data_e

    Inputs:
        conn: connectivity, (nelems, nnodes_per_elem)
        data: nodal data, (nnodes, ) or (nnodes, N)

    Outputs:
        data_e: element-wise data, (nelems, nnodes_per_elem) or (nelems,
                nnodes_per_elem, N)
    """
    data_e[...] = data[conn]
    return


@time_this
def compute_jtrans(Xe, Nderiv, Jq):
    """
    Compute the Jacobian transformation

    Inputs:
        Xe: element nodal location, (nelems, nnodes_per_elem, ndims)
        Nderiv: derivative of basis w.r.t. local coordinate, (nquads,
                nnodes_per_elem, ndims)

    Outputs:
        Jq: Jacobian matrix on quadrature, (nelems, nquads, ndims, ndims)
    """
    Jq[:, :, :, :] = np.einsum("qlk, ilj -> iqjk", Nderiv, Xe)
    return


@time_this
def compute_jdet(Jq, detJq):
    """
    Compute the determinant given Jacobian.

    Inputs:
        Jq: Jacobian matrix on quadrature, (nelems, nquads, ndims, ndims)

    Outputs:
        detJq: Jacobian determinant on quadrature, (nelems, nquads)
    """
    detJq[:, :] = np.linalg.det(Jq)
    return


@time_this
def compute_elem_interp(N, data_e, data_q):
    """
    Interpolate the (scalar or vector) quantities from node to quadrature point
    data_e -> data_q

    Inputs:
        N: basis values, (nquads, nnodes_per_elem)
        data_e: element data, (nelems, nnodes_per_elem) or (nelems,
                nnodes_per_elem, N)

    Outputs:
        data_q: quadrature data, (nelems, nquads) or (nelems, nquads, N)
    """
    if len(data_e.shape) == 2:
        data_q[:, :] = np.einsum("jl, il -> ij", N, data_e)
    else:
        data_q[:, :, :] = np.einsum("jl, ilk -> ijk", N, data_e)
    return


@time_this
def compute_basis_grad(Jq, detJq, Nderiv, invJq, Ngrad):
    """
    Compute the derivatives of basis function with respect to global
    coordinates on quadratures

    Inputs:
        Jq: Jacobian transformation on quadrature, (nelems, nquads, ndims,
            ndims)
        detJq: the determinant of the Jacobian matrices, (nelems, nquads)
        Nderiv: shape function derivatives, (nquads, nnodes_per_elem, ndims)

    Outputs:
        invJq: Jacobian inverse, by-product, (nelems, nquads, ndims, ndims)
        Ngrad: shape function gradient, (nelems, nquads, nnodes_per_elem,
                ndims)
    """
    # Compute Jacobian inverse
    ndims = Jq.shape[-1]
    if ndims == 2:
        invJq[..., 0, 0] = Jq[..., 1, 1] / detJq
        invJq[..., 0, 1] = -Jq[..., 0, 1] / detJq
        invJq[..., 1, 0] = -Jq[..., 1, 0] / detJq
        invJq[..., 1, 1] = Jq[..., 0, 0] / detJq
    else:
        raise NotImplementedError
    Ngrad[:, :, :, :] = np.einsum("jkm, ijml -> ijkl", Nderiv, invJq)
    return


@time_this
def create_dof(nnodes, nelems, nnodes_per_elem, ndof_per_node, nodes, conn):
    """
    Compute dof, dof_each_node and conn_dof

    Inputs:
        nnodes
        nelems
        nnodes_per_elem
        ndof_per_node
        nodes
        conn

    Return:
        dof: the dof indices, (nnodes * ndof_per_node, )
        dof_each_node: the reshaped dof, (nnodes, ndof_per_node)
        conn_dof: nodal dof for each element, (nelems, nnodes_per_elem * ndof_per_node)
    """
    if ndof_per_node == 1:
        dof = nodes
        dof_each_node = nodes
        conn_dof = conn
    else:
        dof = np.zeros((nnodes * ndof_per_node), dtype=int)
        dof_each_node = np.zeros((nnodes, ndof_per_node), dtype=int)
        conn_dof = np.zeros((nelems, nnodes_per_elem * ndof_per_node), dtype=int)
        for axis in range(ndof_per_node):
            dof[axis::ndof_per_node] = axis + ndof_per_node * nodes
            dof_each_node[:, axis] = axis + ndof_per_node * nodes
            conn_dof[:, axis::ndof_per_node] = axis + ndof_per_node * conn

    return dof, dof_each_node, conn_dof


def to_vtk(nodes, conn, X, nodal_sol={}, vtk_name="problem.vtk"):
    """
    Generate a vtk given nodes, conn, X, and optionally nodal_sol

    Inputs:
        nnodes: ndarray
        conn: ndarray or dictionary if a mixed mesh is used
        X: ndarray
        nodal_sol: nodal solution values, dictionary with the following
                    structure:

                    nodal_sol = {
                        "scalar1": [...],
                        "scalar2": [...],
                        ...
                    }

        vtk_name: name of the vtk
    """
    ELEMENT_INFO = {
        "CPS3": {
            "nnode": 3,
            "vtk_type": 5,
            "note": "Three-node plane stress element",
        },
        "C3D8R": {
            "nnode": 8,
            "vtk_type": 12,
            "note": "general purpose linear brick element",
        },
        "C3D10": {
            "nnode": 10,
            "vtk_type": 24,
            "note": "Ten-node tetrahedral element",
        },
        "tri": {
            "nnode": 3,
            "vtk_type": 5,
            "note": "triangle element",
        },
        "quad": {
            "nnode": 4,
            "vtk_type": 9,
            "note": "2d quadrilateral element",
        },
        "block": {
            "nnode": 8,
            "vtk_type": 12,
            "note": "3d block element",
        },
    }

    if isinstance(conn, np.ndarray):
        if conn.shape[1] == 3:
            conn = {"tri": conn}
        elif conn.shape[1] == 4:
            conn = {"quad": conn}
        elif conn.shape[1] == 8:
            conn = {"block": conn}

    # vtk requires a 3-dimensional data point
    if X.shape[1] == 2:
        X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)

    nnodes = len(nodes)
    nelems = np.sum([len(c) for c in conn.values()])

    # Create a empty vtk file and write headers
    with open(vtk_name, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("my example\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")

        # Write nodal points
        fh.write("POINTS {:d} double\n".format(nnodes))
        for x in X:
            row = f"{x}"[1:-1]  # Remove square brackets in the string
            fh.write(f"{row}\n")

        # Write connectivity
        size = np.sum(
            [
                len(econn) * (1 + ELEMENT_INFO[etype]["nnode"])
                for etype, econn in conn.items()
            ]
        )
        fh.write(f"CELLS {nelems} {size}\n")
        for etype, econn in conn.items():
            for c in econn:
                node_idx = f"{c}"[1:-1]  # remove square bracket [ and ]
                npts = ELEMENT_INFO[etype]["nnode"]
                fh.write(f"{npts} {node_idx}\n")

        # Write cell type
        fh.write(f"CELL_TYPES {nelems}\n")
        for etype, econn in conn.items():
            for c in econn:
                vtk_type = ELEMENT_INFO[etype]["vtk_type"]
                fh.write(f"{vtk_type}\n")

        # Write solution
        if nodal_sol:
            fh.write(f"POINT_DATA {nnodes}\n")
            for name, data in nodal_sol.items():
                fh.write(f"SCALARS {name} float 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")
    print(f"[Info] Done generating {vtk_name}")
    return
