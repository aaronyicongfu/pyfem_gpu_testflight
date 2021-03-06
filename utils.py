"""
Utility functions for profiling, data scattering, coordinate transformation, etc.
"""
from time import perf_counter_ns
import numpy as np
import os


class MyProfiler:
    counter = 0  # a static variable
    timer_is_on = True
    print_to_stdout = False
    buffer = []
    istart = []  # stack of indices of open parantheses
    pairs = {}
    t_min = 1  # unit: ms
    log_name = "profiler.log"
    old_log_removed = False
    saved_times = {}

    @staticmethod
    def timer_set_threshold(t: float):
        """
        Don't show entries with elapse time smaller than this. Unit: ms
        """
        MyProfiler.t_min = t
        return

    @staticmethod
    def timer_to_stdout():
        """
        print the profiler output to stdout, otherwise save it as a file
        """
        MyProfiler.print_to_stdout = True
        return

    @staticmethod
    def timer_on():
        """
        Call this function before execution to switch on the profiler
        """
        MyProfiler.timer_is_on = True
        return

    @staticmethod
    def timer_off():
        """
        Call this function before execution to switch off the profiler
        """
        MyProfiler.timer_is_on = False
        return

    @staticmethod
    def time_this(func):
        """
        Decorator: time the execution of a function
        """
        tab = "    "
        fun_name = func.__qualname__

        if not MyProfiler.timer_is_on:

            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)
                return ret

            return wrapper

        def wrapper(*args, **kwargs):
            info_str = f"{tab*MyProfiler.counter}{fun_name}() called"
            entry = {"msg": f"[timer] {info_str:<40s}", "type": "("}
            MyProfiler.buffer.append(entry)

            MyProfiler.counter += 1
            t0 = perf_counter_ns()
            ret = func(*args, **kwargs)
            t1 = perf_counter_ns()
            t_elapse = (t1 - t0) / 1e6  # unit: ms
            MyProfiler.counter -= 1

            info_str = f"{tab*MyProfiler.counter}{fun_name}() return"
            entry = {
                "msg": f"[timer] {info_str:<80s} ({t_elapse:.2f} ms)",
                "type": ")",
                "fun_name": fun_name,
                "t": t_elapse,
            }
            MyProfiler.buffer.append(entry)

            # Once the most outer function returns, we filter the buffer such
            # that we only keep entry pairs whose elapse time is above threshold
            if MyProfiler.counter == 0:
                for idx, entry in enumerate(MyProfiler.buffer):
                    if entry["type"] == "(":
                        MyProfiler.istart.append(idx)
                    if entry["type"] == ")":
                        try:
                            start_idx = MyProfiler.istart.pop()
                            if entry["t"] > MyProfiler.t_min:
                                MyProfiler.pairs[start_idx] = idx
                        except IndexError:
                            print("[Warning]Too many return message")

                # Now our stack should be empty, otherwise we have unpaired
                # called/return message
                if MyProfiler.istart:
                    print("[Warning]Too many called message")

                # Now, we only keep the entries for expensive function calls
                idx = list(MyProfiler.pairs.keys()) + list(MyProfiler.pairs.values())
                if idx:
                    idx.sort()
                keep_buffer = [MyProfiler.buffer[i] for i in idx]

                if MyProfiler.print_to_stdout:
                    for entry in keep_buffer:
                        print(entry["msg"])
                else:
                    if (
                        os.path.exists(MyProfiler.log_name)
                        and not MyProfiler.old_log_removed
                    ):
                        os.remove(MyProfiler.log_name)
                        MyProfiler.old_log_removed = True
                    with open(MyProfiler.log_name, "a") as f:
                        for entry in keep_buffer:
                            f.write(entry["msg"] + "\n")

                # Save time information to dictionary
                for entry in keep_buffer:
                    if "t" in entry.keys():
                        _fun_name = entry["fun_name"]
                        _t = entry["t"]
                        if _fun_name in MyProfiler.saved_times.keys():
                            MyProfiler.saved_times[_fun_name].append(_t)
                        else:
                            MyProfiler.saved_times[_fun_name] = [_t]

                # Reset buffer and pairs
                MyProfiler.buffer = []
                MyProfiler.pairs = {}
            return ret

        return wrapper


time_this = MyProfiler.time_this
timer_on = MyProfiler.timer_on
timer_off = MyProfiler.timer_off
timer_to_stdout = MyProfiler.timer_to_stdout
timer_set_threshold = MyProfiler.timer_set_threshold


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
        # fmt: off
        invJq[..., 0, 0] =  (Jq[..., 1, 1] * Jq[..., 2, 2] - Jq[..., 1, 2] * Jq[..., 2, 1]) / detJq
        invJq[..., 0, 1] = -(Jq[..., 0, 1] * Jq[..., 2, 2] - Jq[..., 0, 2] * Jq[..., 2, 1]) / detJq
        invJq[..., 0, 2] =  (Jq[..., 0, 1] * Jq[..., 1, 2] - Jq[..., 0, 2] * Jq[..., 1, 1]) / detJq

        invJq[..., 1, 0] = -(Jq[..., 1, 0] * Jq[..., 2, 2] - Jq[..., 1, 2] * Jq[..., 2, 0]) / detJq
        invJq[..., 1, 1] =  (Jq[..., 0, 0] * Jq[..., 2, 2] - Jq[..., 0, 2] * Jq[..., 2, 0]) / detJq
        invJq[..., 1, 2] = -(Jq[..., 0, 0] * Jq[..., 1, 2] - Jq[..., 0, 2] * Jq[..., 1, 0]) / detJq

        invJq[..., 2, 0] =  (Jq[..., 1, 0] * Jq[..., 2, 1] - Jq[..., 1, 1] * Jq[..., 2, 0]) / detJq
        invJq[..., 2, 1] = -(Jq[..., 0, 0] * Jq[..., 2, 1] - Jq[..., 0, 1] * Jq[..., 2, 0]) / detJq
        invJq[..., 2, 2] =  (Jq[..., 0, 0] * Jq[..., 1, 1] - Jq[..., 0, 1] * Jq[..., 1, 0]) / detJq
        # fmt: on

    Ngrad[:, :, :, :] = np.einsum("jkm, ijml -> ijkl", Nderiv, invJq)
    return


@time_this
def create_dof(nnodes, nelems, nnodes_per_elem, ndof_per_node, conn):
    """
    Compute dof, dof_each_node and conn_dof

    Inputs:
        nnodes
        nelems
        nnodes_per_elem
        ndof_per_node
        conn

    Return:
        dof: the dof indices, (nnodes * ndof_per_node, )
        dof_each_node: the reshaped dof, (nnodes, ndof_per_node)
        conn_dof: nodal dof for each element, (nelems, nnodes_per_elem * ndof_per_node)
    """
    nodes = np.arange(nnodes)
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


@time_this
def to_vtk(conn, X, nodal_sol={}, vtk_name="problem.vtk"):
    """
    Generate a vtk given conn, X, and optionally nodal_sol

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
        "tet": {
            "nnode": 10,
            "vtk_type": 24,
            "note": "Ten-node tetrahedral element",
        },
        "brick20": {
            "nnode": 20,
            "vtk_type": 12,
            "note": "20-node brick element",
        }
    }

    if isinstance(conn, np.ndarray):
        if conn.shape[1] == 3:
            conn = {"tri": conn}
        elif conn.shape[1] == 4:
            conn = {"quad": conn}
        elif conn.shape[1] == 8:
            conn = {"block": conn}
        elif conn.shape[1] == 10:
            conn = {"tet": conn}
        elif conn.shape[1] == 20:
            conn = {"brick20": conn}

    # vtk requires a 3-dimensional data point
    if X.shape[1] == 2:
        X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)

    nnodes = X.shape[0]
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
