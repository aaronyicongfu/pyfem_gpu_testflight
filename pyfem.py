from re import A
import numpy as np
from time import perf_counter_ns, time
from abc import ABC, abstractmethod


def time_this(func):
    """
    Decorator: time the execution of a function
    """
    def wrapper(*args, **kwargs):
        t0 = perf_counter_ns()
        ret = func(*args, **kwargs)
        t1 = perf_counter_ns()
        print(f'[timer] {func.__name__:s}() returns,',
              f'exec time: {(t1 - t0) / 1e6:.2f} ms')
        return ret
    return wrapper


class QuadratureBase(ABC):
    """
    Abstract class for quadrature object
    """
    def __init__(self, pts, weights):
        self.pts = pts
        self.weights = weights
        self.num_quad_pts = pts.shape[0]
        return

    def get_num_quad_pts(self):
        """
        Get number of quadrature points per element
        """
        return self.num_quad_pts

    @abstractmethod
    def get_pt(self, idx=None):
        """
        Query the <idx>-th quadrature point (xi, eta) or (xi, eta, zeta) based
        on quadrature type, if idx is None, return all quadrature points as a
        list
        """
        if idx:
            return self.pts[idx]
        else:
            return self.pts

    @abstractmethod
    def get_weight(self, idx=None):
        """
        Query the weight of <idx>-th quadrature point, if idx is None, return
        all quadrature points as a list
        """
        if idx:
            return self.weights[idx]
        else:
            return self.weights


class BasisBase(ABC):
    """
    Abstract class for the element basis function
    """
    @abstractmethod
    def eval_shape_fun(self, quad_pts, N):
        """
        Evaluate the shape function values at each quadrature point

        Outputs:
            N: shape function values of shape (num_quad_pts, nnodes_per_elem)
        """
        return

    @abstractmethod
    def eval_shape_fun_deriv(self, quad_pts, Nderiv):
        """
        Evaluate the shape function derivatives at each quadrature point

        Outputs:
            Nderiv: shape function values of shape (num_quad_pts,
            nnodes_per_elem, ndim)
        """
        return


class PhysicalModelBase(ABC):
    """
    Abstract class for the physical problem to be solved by the finite element
    method
    TODO: how to generalize this class?
    """
    def __init__(self, nvars_per_node):
        self.nvars_per_node = nvars_per_node
        return

    def get_nvars_per_node(self):
        return self.nvars_per_node

    @abstractmethod
    def eval_element_rhs(self):
        return

    @abstractmethod
    def eval_element_jacobian(self):
        return


class Bilinear2DQuadrature(QuadratureBase):
    def __init__(self):
        pts = np.array([[-1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)],
                        [ 1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)],
                        [ 1.0/np.sqrt(3.0),  1.0/np.sqrt(3.0)],
                        [-1.0/np.sqrt(3.0),  1.0/np.sqrt(3.0)]])
        weights = np.array([1., 1., 1., 1.])
        super().__init__(pts, weights)
        return

    def get_pt(self, idx=None):
        return super().get_pt(idx)

    def get_weight(self, idx=None):
        return super().get_weight(idx)


class Bilinear2DBasis(BasisBase):
    def eval_shape_fun(self, quad_pts, N):
        N[...] = np.array(
            list(
                map(
                    lambda qpt:
                        [0.25*(1.0 - qpt[0])*(1.0 - qpt[1]),
                         0.25*(1.0 + qpt[0])*(1.0 - qpt[1]),
                         0.25*(1.0 + qpt[0])*(1.0 + qpt[1]),
                         0.25*(1.0 - qpt[0])*(1.0 + qpt[1])],
                    quad_pts
                    )
                )
            )
        return

    def eval_shape_fun_deriv(self, quad_pts, Nderiv):
        Nderiv[..., 0] = np.array(
            list(
                map(
                    lambda qpt:
                        [-0.25*(1.0 - qpt[1]),
                          0.25*(1.0 - qpt[1]),
                          0.25*(1.0 + qpt[1]),
                         -0.25*(1.0 + qpt[1])],
                    quad_pts
                    )
                )
            )
        Nderiv[..., 1] = np.array(
            list(
                map(
                    lambda qpt:
                        [-0.25*(1.0 - qpt[0]),
                         -0.25*(1.0 + qpt[0]),
                          0.25*(1.0 + qpt[0]),
                          0.25*(1.0 - qpt[0])],
                    quad_pts
                    )
                )
            )
        return


class LinearPoisson2D(PhysicalModelBase):
    """
    The 2-dimensional linear poisson physics
    """
    def __init__(self):
        self.fun_vals = None
        nvars_per_node = 1
        super().__init__(nvars_per_node)
        return

    def _eval_gfun(self, Xq, fun_vals):
        xvals = Xq[..., 0]
        yvals = Xq[..., 1]
        fun_vals[...] =  xvals * (xvals - 5.0) * (xvals - 10.0) * yvals * (yvals - 4.0)
        return

    def eval_element_rhs(self, quad_weights, N, detJq, Xq, rhs_e):
        """
        Evaluate element-wise rhs vectors

        Inputs:
            quad_weights: quadrature weights of shape (num_quad_pts, )
            N: shape function values of shape (nnodes_per_elem, num_quad_pts)
            detJq: Jacobian determinants of shape (nelems, num_quad_pts)
            Xq: quadrature location in global coordinates of shape (nelems,
                num_quad_pts, ndim)

        Outputs:
            rhs_e: element-wise rhs of shape (nelems, nnodes_per_elem)
        """
        self._eval_gfun(Xq, self.fun_vals)
        rhs_e[...] = np.einsum('k,jk,ik,ik -> ijk',
            quad_weights, N, detJq, self.fun_vals).sum(axis=-1)
        return

    def eval_element_jacobian(self):
        return


class Assembler:
    def __init__(self, conn, X, quadrature, basis, model):
        # Prepare and take inputs
        self.conn = np.array(conn)  # connectivity: nelems * nnodes_per_elem
        self.X = np.array(X)  # nodal locations: nnodes * ndim
        self.quadrature = quadrature
        self.basis = basis
        self.model = model  # physical model

        # Get dimension information
        self.nelems = conn.shape[0]
        self.nnodes_per_elem = conn.shape[1]
        self.ndim = X.shape[1]
        self.num_quad_pts = self.quadrature.get_num_quad_pts()
        self.nvars_per_node = self.model.get_nvars_per_node()  # how many variables for each node

        # Allocate memory for the element-wise data
        self.Xe = np.zeros((self.nelems, self.nnodes_per_elem,
                            self.ndim))  # nodal locations
        self.Ue = np.zeros((self.nelems, self.nnodes_per_elem,
                            self.nvars_per_node))  # solution variable
        self.rhs_e = np.zeros((self.nelems,
                               self.nnodes_per_elem * \
                               self.nvars_per_node))  # rhs vectors for each
                                                      # element

        # Allocate memory for quadrature data
        self.Xq = np.zeros((self.nelems, self.num_quad_pts,
                            self.ndim))  # nodal locations
        self.Jq = np.zeros((self.nelems, self.num_quad_pts,
                               self.ndim, self.ndim))  # Jacobian transformations
        self.invJq = np.zeros((self.nelems, self.num_quad_pts,
                               self.ndim, self.ndim))  # inverse of Jacobian
                                                       # transformations
        self.detJq = np.zeros((self.nelems,
                               self.num_quad_pts))  # determinants of Jacobian
                                                    # transformations
        self.N = np.zeros((self.num_quad_pts,
                           self.nnodes_per_elem))  # shape function values
        self.Nderiv = np.zeros((self.num_quad_pts,
                                 self.nnodes_per_elem,
                                 self.ndim))  # shape function derivatives
                                              # w.r.t. local coordinates

        # Global Jacobian (stiffness) matrix and rhs to be created
        self.K = None
        self.rhs = None
        return

    @time_this
    def _compute_jtrans(self, Xe, Nderiv, Jq, detJq, invJq):
        """
        Compute the Jacobian transformation, inverse and determinant

        Outputs:
            Jq: the Jacobian matrices for each element and each quadrature point
            detJq: the determinant of the Jacobian matrices
            invJq: the inverse of the Jacobian matrices
        """
        # Compute the Jacobian transformations
        Jq[...] = np.einsum('ijk, mjn -> imkn', Xe, Nderiv)

        if self.ndim == 2:
            # Compute the determinant and inverse
            detJq[...] = Jq[..., 0, 0] * Jq[..., 1, 1] \
                       - Jq[..., 0, 1] * Jq[..., 1, 0]
            invJq[..., 0, 0] =  Jq[..., 1, 1] / detJq
            invJq[..., 0, 1] = -Jq[..., 0, 1] / detJq
            invJq[..., 1, 0] = -Jq[..., 1, 0] / detJq
            invJq[..., 1, 1] =  Jq[..., 0, 0] / detJq

        elif self.ndim == 1:
            #TODO: Is this even needed?
            raise NotImplementedError()
        elif self.ndim == 3:
            #TODO: Implement the 3-dimensional case
            raise NotImplementedError()
        return

    @time_this
    # TODO: rename this to interp_quad_vals() ?
    # def _compute_elem_interp(self, data_e, N, data_q):
    def _compute_elem_mapping(self, data_e, N, data_q):
        """
        Interpolate the quantities at quadrature points:
        data_e -> data_q

        Inputs:
            data_e: element data of shape (nelems, nnodes_per_elem, N)
            N: basis values of shape (num_quad_pts, nnodes_per_elem)

        Outputs:
            data_q: quadrature data of shape (nelems, num_quad_pts, N)
        """
        data_q[...] = np.einsum('jl, ilk -> ijk', N, data_e)
        return

    # TODO:
    # Maybe use this to compute the Jacobian
    def _compute_elem_grad(self, data_e, Nderiv, data_q):
        return

    @time_this
    def _assemble_rhs(self, conn, quadrature, rhs_e, rhs):
        """
        Assemble the global rhs given element-wise rhs: rhs_e -> rhs

        Inputs:
            conn: connectivity
            quadrature: the quadrature object
            rhs_e: element-wise rhs of shape (nelems, nnodes_per_elem *
                   nvars_per_node)

        Outputs:
            rhs: global rhs of shape (nnodes * nvars_per_node, )
        """
        for n in range(quadrature.get_num_quad_pts()):
            np.add.at(rhs[:], conn[:, n], rhs_e[:, n])
        return

    @time_this
    def compute_rhs(self, rhs):
        """
        Compute the element rhs vectors and assemble the global rhs
        Outputs:
            rhs: global right-hand-side vector
        """
        # Compute shape function and derivatives
        quad_pts = self.quadrature.get_pt()
        quad_weights = self.quadrature.get_weight()
        self.basis.eval_shape_fun(quad_pts, self.N)
        self.basis.eval_shape_fun_deriv(quad_pts, self.Nderiv)

        # Compute Jacobian transformation -> Jq, detJq, invJq
        self._compute_jtrans(self.Xe, self.Nderiv, self.Jq, self.detJq,
                             self.invJq)

        # Compute element mapping -> Xq
        self._compute_elem_mapping(self.Xe, self.N, self.Xq)

        # Compute element rhs vector -> rhs_e
        self.model.eval_element_rhs(quad_weights, self.N, self.detJq,
                                    self.Xq, self.rhs_e)

        # Assemble the global rhs vector -> rhs
        self._assemble_rhs(self.conn, self.quadrature, self.rhs_e, rhs)
        return

    def compute_jacobian(self):
        return


def create_problem(n=10):
    m = n * 4
    nelems = m*n
    nnodes = (m + 1)*(n + 1)
    y = np.linspace(0, 4, n + 1)
    x = np.linspace(0, 10, m + 1)
    nodes = np.arange(0, (n + 1)*(m + 1)).reshape((n + 1, m + 1))

    # Set the node locations
    X = np.zeros((nnodes, 2))
    for j in range(n + 1):
        for i in range(m + 1):
            X[i + j*(m + 1), 0] = x[i]
            X[i + j*(m + 1), 1] = y[j]

    # Set the connectivity
    conn = np.zeros((nelems, 4), dtype=int)
    for j in range(n):
        for i in range(m):
            conn[i + j*m, 0] = nodes[j, i]
            conn[i + j*m, 1] = nodes[j, i + 1]
            conn[i + j*m, 2] = nodes[j + 1, i + 1]
            conn[i + j*m, 3] = nodes[j + 1, i]

    return conn, X, nelems, nnodes


if __name__ == '__main__':
    conn, X, nelems, nnodes = create_problem()
    quadrature = Bilinear2DQuadrature()
    basis = Bilinear2DBasis()
    model = LinearPoisson2D()
    assembler = Assembler(conn, X, quadrature, basis, model)
    rhs = np.zeros(nnodes)
    assembler.compute_rhs(rhs)
