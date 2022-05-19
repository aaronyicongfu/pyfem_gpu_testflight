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
