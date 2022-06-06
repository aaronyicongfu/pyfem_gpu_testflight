import numpy as np
from scipy import sparse
from scipy import special
from scipy.sparse.linalg import spsolve, gmres, cg
from abc import ABC, abstractmethod
import matplotlib.tri as tri
import pyamg
import utils
from utils import time_this
from typing import Callable


class QuadratureBase(ABC):
    """
    Abstract base class for quadrature object
    """

    @abstractmethod
    def __init__(self, pts, weights):
        """
        Args:
            pts: list-like, list of quadrature points
            weights: list-like, list of quadrature weights
        """
        assert len(pts) == len(weights)  # sanity check
        self.pts = pts
        self.weights = weights
        self.nquads = pts.shape[0]
        return

    @time_this
    def get_nquads(self):
        """
        Get number of quadrature points per element
        """
        return self.nquads

    @time_this
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

    @time_this
    def get_weight(self, idx=None):
        """
        Query the weight of <idx>-th quadrature point, if idx is None, return
        all quadrature points as a list
        """
        if idx:
            return self.weights[idx]
        else:
            return self.weights


class QuadratureTriangle2D(QuadratureBase):
    """
    Linear triangular element only has one quadrature point (L1=1/3, L2=1/3)
    """

    @time_this
    def __init__(self):
        pts = np.array([[1.0 / 3, 1.0 / 3]])
        area_element_local_coord = 0.5
        weights = np.array([1.0])
        weights *= area_element_local_coord  # TODO: I think numbering order also matters here...
        super().__init__(pts, weights)
        return


class QuadratureBilinear2D(QuadratureBase):
    @time_this
    def __init__(self):
        # fmt: off
        pts = np.array([[-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [ 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [ 1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
                        [-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)]])
        # fmt: on
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        super().__init__(pts, weights)
        return


class QuadratureBlock3D(QuadratureBase):
    @time_this
    def __init__(self):
        # fmt: off
        pts = np.array([[-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
                        [-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
                        [+1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [+1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
                        [+1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [+1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)]])
        # fmt: on
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        super().__init__(pts, weights)
        return


class BasisBase(ABC):
    """
    Abstract class for the element basis function
    """

    @abstractmethod
    def __init__(self, ndims, nnodes_per_elem, quadrature: QuadratureBase):
        """
        Inputs:
            ndims: int, number of physical dimensions (2 or 3)
            nnodes_per_elem: int, number of nodes per element
            quadrature: object of type QuadratureBase
        """
        self.ndims = ndims
        self.nnodes_per_elem = nnodes_per_elem
        self.quadrature = quadrature
        self.nquads = self.quadrature.get_nquads()
        self.N = None
        self.Nderiv = None
        return

    @abstractmethod
    def _eval_shape_fun_on_quad_pt(self, qpt):
        """
        Args:
            qpt: a single quadrature point in local coordinate (xi, eta, ...)

        Return:
            shape_vals: list-like, shape function value for each node
        """
        shape_vals = []
        return shape_vals

    @abstractmethod
    def _eval_shape_deriv_on_quad_pt(self, qpt):
        """
        Args:
            qpt: a single quadrature point in local coordinate (xi, eta, ...)

        Return:
            shape_derivs: 1-dim list, [*dN1, *dN2, ...], where dNi = [dNi/dxi,
                          dNi/deta, ..]
        """
        shape_derivs = []
        return shape_derivs

    @time_this
    def eval_shape_fun(self):
        """
        Evaluate the shape function values at quadrature points

        Return:
            shape function values at each quadrature point
        """
        if self.N is None:
            self.N = np.zeros((self.nquads, self.nnodes_per_elem))
            self.N[:, :] = list(
                map(self._eval_shape_fun_on_quad_pt, self.quadrature.get_pt())
            )
        return self.N

    @time_this
    def eval_shape_fun_deriv(self):
        """
        Evaluate the shape function derivatives at quadrature points

        Return:
            shape function derivatives at each quadrature point w.r.t. each
            local coordinate xi, eta, ...
        """
        if self.Nderiv is None:
            self.Nderiv = np.zeros((self.nquads, self.nnodes_per_elem, self.ndims))
            self.Nderiv[:, :, :] = np.array(
                list(map(self._eval_shape_deriv_on_quad_pt, self.quadrature.get_pt()))
            ).reshape((self.nquads, self.nnodes_per_elem, self.ndims))
        return self.Nderiv


class BasisBilinear2D(BasisBase):
    @time_this
    def __init__(self, quadrature: QuadratureBase):
        ndims = 2
        nnodes_per_elem = 4
        super().__init__(ndims, nnodes_per_elem, quadrature)
        return

    @time_this
    def _eval_shape_fun_on_quad_pt(self, qpt):
        shape_vals = [
            0.25 * (1.0 - qpt[0]) * (1.0 - qpt[1]),
            0.25 * (1.0 + qpt[0]) * (1.0 - qpt[1]),
            0.25 * (1.0 + qpt[0]) * (1.0 + qpt[1]),
            0.25 * (1.0 - qpt[0]) * (1.0 + qpt[1]),
        ]
        return shape_vals

    @time_this
    def _eval_shape_deriv_on_quad_pt(self, qpt):
        shape_derivs = [
            # fmt: off
            -0.25 * (1.0 - qpt[1]),
            -0.25 * (1.0 - qpt[0]),
             0.25 * (1.0 - qpt[1]),
            -0.25 * (1.0 + qpt[0]),
             0.25 * (1.0 + qpt[1]),
             0.25 * (1.0 + qpt[0]),
            -0.25 * (1.0 + qpt[1]),
             0.25 * (1.0 - qpt[0]),
        ]
        return shape_derivs


class BasisBlock3D(BasisBase):
    @time_this
    def __init__(self, quadrature: QuadratureBase):
        ndims = 3
        nnodes_per_elem = 8
        super().__init__(ndims, nnodes_per_elem, quadrature)
        return

    @time_this
    def _eval_shape_fun_on_quad_pt(self, qpt):
        shape_vals = [
            0.125 * (1.0 - qpt[0]) * (1.0 - qpt[1]) * (1.0 - qpt[2]),
            0.125 * (1.0 + qpt[0]) * (1.0 - qpt[1]) * (1.0 - qpt[2]),
            0.125 * (1.0 + qpt[0]) * (1.0 + qpt[1]) * (1.0 - qpt[2]),
            0.125 * (1.0 - qpt[0]) * (1.0 + qpt[1]) * (1.0 - qpt[2]),
            0.125 * (1.0 - qpt[0]) * (1.0 - qpt[1]) * (1.0 + qpt[2]),
            0.125 * (1.0 + qpt[0]) * (1.0 - qpt[1]) * (1.0 + qpt[2]),
            0.125 * (1.0 + qpt[0]) * (1.0 + qpt[1]) * (1.0 + qpt[2]),
            0.125 * (1.0 - qpt[0]) * (1.0 + qpt[1]) * (1.0 + qpt[2]),
        ]
        return shape_vals

    @time_this
    def _eval_shape_deriv_on_quad_pt(self, qpt):
        shape_derivs = [
            # fmt: off
            -0.125 * (1.0 - qpt[1]) * (1.0 - qpt[2]),
            -0.125 * (1.0 - qpt[0]) * (1.0 - qpt[2]),
            -0.125 * (1.0 - qpt[0]) * (1.0 - qpt[1]),
             0.125 * (1.0 - qpt[1]) * (1.0 - qpt[2]),
            -0.125 * (1.0 + qpt[0]) * (1.0 - qpt[2]),
            -0.125 * (1.0 + qpt[0]) * (1.0 - qpt[1]),
             0.125 * (1.0 + qpt[1]) * (1.0 - qpt[2]),
             0.125 * (1.0 + qpt[0]) * (1.0 - qpt[2]),
            -0.125 * (1.0 + qpt[0]) * (1.0 + qpt[1]),
            -0.125 * (1.0 + qpt[1]) * (1.0 - qpt[2]),
             0.125 * (1.0 - qpt[0]) * (1.0 - qpt[2]),
            -0.125 * (1.0 - qpt[0]) * (1.0 + qpt[1]),
            -0.125 * (1.0 - qpt[1]) * (1.0 + qpt[2]),
            -0.125 * (1.0 - qpt[0]) * (1.0 + qpt[2]),
             0.125 * (1.0 - qpt[0]) * (1.0 - qpt[1]),
             0.125 * (1.0 - qpt[1]) * (1.0 + qpt[2]),
            -0.125 * (1.0 + qpt[0]) * (1.0 + qpt[2]),
             0.125 * (1.0 + qpt[0]) * (1.0 - qpt[1]),
             0.125 * (1.0 + qpt[1]) * (1.0 + qpt[2]),
             0.125 * (1.0 + qpt[0]) * (1.0 + qpt[2]),
             0.125 * (1.0 + qpt[0]) * (1.0 + qpt[1]),
            -0.125 * (1.0 + qpt[1]) * (1.0 + qpt[2]),
             0.125 * (1.0 - qpt[0]) * (1.0 + qpt[2]),
             0.125 * (1.0 - qpt[0]) * (1.0 + qpt[1]),
        ]
        return shape_derivs


class BasisTriangle2D(BasisBase):
    """
    Linear triangular element has 3 area coordinates L1, L2, L3 with the
    constraint L1 + L2 + L3 = 1, hence we only consider the first two
    independent coordinates as local coordinates.

    Shape function:
        N = [N1, N2, N3]
        N1 = L1
        N2 = L2
        N3 = 1 - L1 - L2

    Coordinate transformation:
        x = L1 * x1 + L2 * x2 + (1 - L1 - L2) * x3
        y = L1 * y1 + L2 * y2 + (1 - L1 - L2) * y3

    Jacobian transformation:
        [dx/dL1, dx/dL2  = [x1 - x3, x2 - x3
         dy/dL1, dy/dL2]   [y1 - y3, y2 - y3]
    """

    @time_this
    def __init__(self, quadrature: QuadratureBase):
        ndims = 2
        nnodes_per_elem = 3
        super().__init__(ndims, nnodes_per_elem, quadrature)
        return

    @time_this
    def _eval_shape_fun_on_quad_pt(self, qpt):
        shape_vals = [qpt[0], qpt[1], 1 - qpt[0] - qpt[1]]
        return shape_vals

    @time_this
    def _eval_shape_deriv_on_quad_pt(self, qpt):
        shape_derivs = [1.0, 0.0, 0.0, 1.0, -1.0, -1.0]
        return shape_derivs


class ModelBase(ABC):
    """
    Abstract base class for the problem model and physics
    """

    @abstractmethod
    def __init__(
        self,
        ndof_per_node,
        X,
        conn,
        dof_fixed,
        dof_fixed_vals,
        quadrature: QuadratureBase,
        basis: BasisBase,
    ):
        """
        ndof_per_node: int, number of component of state variable
        X: nodal location matrix, (nnodes, ndims)
        conn: connectivity matrix, (nelems, nnodes_per_elem)
        dof_fixed: list of dof indices to enforce boundary condition, (Ndof_bc, )
        dof_fixed_vals: list of bc values, (Ndof_bc, ), None means all fixed
                        values are 0
        """
        self.ndof_per_node = ndof_per_node
        self.X = np.array(X, dtype=float)
        self.conn = np.array(conn, dtype=int)
        self.dof_fixed = np.array(dof_fixed, dtype=int)
        if dof_fixed_vals is None:
            self.dof_fixed_vals = None
        else:
            self.dof_fixed_vals = np.array(dof_fixed_vals, dtype=float)
        self.quadrature = quadrature
        self.basis = basis

        # Get dimension information
        self.nelems = conn.shape[0]
        self.nnodes_per_elem = conn.shape[1]
        self.nnodes = X.shape[0]
        self.ndims = X.shape[1]
        self.nquads = quadrature.get_nquads()

        # Set nodal numbering
        self.nodes = np.arange(self.nnodes)

        # Sanity check: conn
        assert self.conn.min() == 0
        assert self.conn.max() == self.nnodes - 1
        assert len(set(self.conn.flatten())) == self.nnodes

        """
        Compute dof information
        """
        # Create dof arrays
        self.dof, self.dof_each_node, self.conn_dof = utils.create_dof(
            self.nnodes,
            self.nelems,
            self.nnodes_per_elem,
            self.ndof_per_node,
            self.conn,
        )

        # Compute free dof indices
        self.dof_free = np.setdiff1d(self.dof, self.dof_fixed)

        """
        Allocate memory for the element-wise data
        """
        # Nodal coordinates
        self.Xe = np.zeros((self.nelems, self.nnodes_per_elem, self.ndims))
        utils.scatter_node_to_elem(self.conn, self.X, self.Xe)

        # Element-wise rhs
        self.rhs_e = np.zeros((self.nelems, self.nnodes_per_elem * self.ndof_per_node))

        # Element-wise Jacobian tensor and matrix
        self.Ke_tensor = np.zeros(
            (
                self.nelems,
                self.nnodes_per_elem,
                self.nnodes_per_elem,
                self.ndof_per_node,
                self.ndof_per_node,
            )
        )
        self.Ke_mat = np.zeros(
            (
                self.nelems,
                self.nnodes_per_elem * self.ndof_per_node,
                self.nnodes_per_elem * self.ndof_per_node,
            )
        )

        """
        Allocate memory for quadrature data
        """
        # Nodal coordinates
        self.Xq = np.zeros((self.nelems, self.nquads, self.ndims))

        # Jacobian transformation
        self.Jq = np.zeros((self.nelems, self.nquads, self.ndims, self.ndims))

        # Jacobian transformation inverse
        self.invJq = np.zeros((self.nelems, self.nquads, self.ndims, self.ndims))

        # Jacobian determinant
        self.detJq = np.zeros((self.nelems, self.nquads))

        # gradient of basis w.r.t. global coordinates
        self.Ngrad = np.zeros(
            (self.nelems, self.nquads, self.nnodes_per_elem, self.ndims)
        )

        """
        Global linear system
        """
        # Compute indices for non-zero entries
        self.nz_i, self.nz_j = self._compute_nz_pattern()

        # rhs and K
        self.rhs = np.zeros(self.nnodes * self.ndof_per_node)
        return

    @abstractmethod
    def compute_rhs(self):
        """
        Compute the global rhs vector (without considering boundary conditions)

        Return:
            self.rhs
        """
        return self.rhs

    @abstractmethod
    def compute_jacobian(self):
        """
        Compute the global Jacobian matrix (without considering boundary conditions)

        Return:
            K
        """
        K = None
        return K

    @time_this
    def apply_dirichlet_bcs(self, K, rhs, enforce_symmetric_K=True):
        """
        Apply Dirichlet boundary conditions to global Jacobian matrix and
        right-hand-side vector. In general, the linear system with boundary
        conditions becomes:

            [Krr Krb  [ur    [fr
             0   I  ]  u0] =  u0]

        where u0 is the fixed values for Dirichlet bc, ur is the unknown states,
        optionally, to make the system symmetric, we could move Krb to rhs:

         => [Krr 0  [ur    [fr - Krb u0
             0   I]  u0] =  u0         ]

        Inputs:
            K: stiffness matrix
            rhs: right-hand-side vector
            enforce_symmetric_K: zero out rows and columns to maintain symmetry

        Return:
            K: stiffness matrix with bcs
            rhs: rhs with bcs
        """
        # Save Krb and diagonals
        temp = K[self.dof_free, :]
        Krb = temp[:, self.dof_fixed]
        diag = K.diagonal()

        # Zero-out rows
        for i in self.dof_fixed:
            K.data[K.indptr[i] : K.indptr[i + 1]] = 0

        # Zero-out columns
        if enforce_symmetric_K:
            K = K.tocsc()
            for i in self.dof_fixed:
                K.data[K.indptr[i] : K.indptr[i + 1]] = 0
            K = K.tocsr()

        # Set diagonals to 1
        diag[self.dof_fixed] = 1.0
        K.setdiag(diag)

        # Remove 0
        K.eliminate_zeros()

        # Set rhs
        if self.dof_fixed_vals is None:
            rhs[self.dof_fixed] = 0.0
        else:
            rhs[self.dof_fixed] = self.dof_fixed_vals[:]
            if enforce_symmetric_K:
                rhs[self.dof_free] -= Krb.dot(self.dof_fixed_vals)
        return K, rhs

    @time_this
    def _compute_nz_pattern(self):
        """
        Compute indices for non-zero entries in the global stiffness matrix

        Return:
            nz_i: i-coordinates
            nz_j: j-coordinates
        """
        elem_to_mat_i = [
            i
            for i in range(self.nnodes_per_elem * self.ndof_per_node)
            for j in range(self.nnodes_per_elem * self.ndof_per_node)
        ]
        elem_to_mat_j = [
            j
            for i in range(self.nnodes_per_elem * self.ndof_per_node)
            for j in range(self.nnodes_per_elem * self.ndof_per_node)
        ]
        nz_i = self.conn_dof[:, elem_to_mat_i].flatten()
        nz_j = self.conn_dof[:, elem_to_mat_j].flatten()
        return nz_i, nz_j

    @time_this
    def _assemble_rhs(self, rhs_e, rhs):
        """
        Assemble the global rhs given element-wise rhs: rhs_e -> rhs

        Inputs:
            quadrature: the quadrature object
            rhs_e: element-wise rhs, (nelems, nnodes_per_elem * ndof_per_node)

        Outputs:
            rhs: global rhs, (nnodes * ndof_per_node, )
        """
        rhs[:] = 0.0
        for n in range(self.quadrature.get_nquads()):
            np.add.at(rhs[:], self.conn_dof[:, n], rhs_e[:, n])
        return

    @time_this
    def _jacobian_mat_to_tensor(self, mat, tensor):
        """
        Convert element Jacobian matrix to tensor

        Inputs:
            mat: (nelems, nnodes_per_elem * ndof_per_node, nnodes_per_elem *
                 ndof_per_node)

        Outputs:
            tensor: (nelems, nnodes_per_elem, nnodes_per_elem, ndof_per_node,
                    ndof_per_node)
        """
        nelems, nnodes_per_elem, _, ndof_per_node, _ = tensor.shape
        tensor[:, :, :, :, :] = (
            mat.reshape(nelems, -1, ndof_per_node, nnodes_per_elem * ndof_per_node)
            .swapaxes(2, 3)
            .reshape(
                nelems, nnodes_per_elem, nnodes_per_elem, ndof_per_node, ndof_per_node
            )
            .swapaxes(3, 4)
        )
        return

    @time_this
    def _jacobian_tensor_to_mat(self, tensor, mat):
        """
        Convert element Jacobian tensor to matrix

        Inputs:
            tensor: (nelems, nnodes_per_elem, nnodes_per_elem, ndof_per_node,
                    ndof_per_node)

        Outputs:
            mat: (nelems, nnodes_per_elem * ndof_per_node, nnodes_per_elem *
                 ndof_per_node)
        """
        nelems, nnodes_per_elem, _, ndof_per_node, _ = tensor.shape
        mat[:, :, :] = tensor.swapaxes(2, 3).reshape(
            nelems, nnodes_per_elem * ndof_per_node, nnodes_per_elem * ndof_per_node
        )
        return

    @time_this
    def _assemble_jacobian(self, Ke_mat):
        """
        Assemble global K matrix from Ke matrices

        Inputs:
            Ke_mat: element-wise Jacobian in matrix form, (nelems,
                    nnodes_per_elem * ndof_per_node, nnodes_per_elem
                    * ndof_per_node)
        """
        K = sparse.coo_matrix((Ke_mat.flatten(), (self.nz_i, self.nz_j)))
        return K.tocsr()


class LinearPoisson(ModelBase):
    """
    The 2- or 3-dimensional Poisson equation

    Equation:
    -k∆u = g in Ω

    Boundary condition:
    u = u0 on ∂Ω

    Weak form:
    ∫ ∇u ∇v dΩ = ∫gv dΩ for test function v
    """

    @time_this
    def __init__(
        self,
        X,
        conn,
        dof_fixed,
        dof_fixed_vals,
        quadrature: QuadratureBase,
        basis: BasisBase,
        gfunc: Callable,
        kappa0=1.0,
        p=0.0,
    ):
        """
        Inputs:
            gfunc: source term, takes takes in x and return vals, where
                   x[..., 0] = xvals
                   x[..., 1] = yvals
                   x[..., 2] = zvals if is 3D problem
                   vals.shape == x[..., 0].shape
            kappa0: the thermal conductivity of the material
            p: RAMP penalization (for topology optimization)
        """
        ndof_per_node = 1
        super().__init__(
            ndof_per_node, X, conn, dof_fixed, dof_fixed_vals, quadrature, basis
        )
        self.gfunc = gfunc
        self.fun_vals = None
        self.kappa0 = kappa0
        self.p = p

        # Allocate memory
        self.rho_e = np.zeros((self.nelems, self.nnodes_per_elem))
        self.rho_q = np.zeros((self.nelems, self.nquads))
        self.kappa_q = np.zeros((self.nelems, self.nquads))
        self.kappa_q_deriv = np.zeros((self.nelems, self.nquads, self.nnodes_per_elem))
        self.inner = np.zeros((self.nelems, self.nnodes_per_elem))
        self.Ke_deriv = np.zeros(
            (
                self.nelems,
                self.nnodes_per_elem * self.ndof_per_node,
                self.nnodes_per_elem * self.ndof_per_node,
                self.nnodes_per_elem,
            )
        )
        return

    @time_this
    def compute_rhs(self):
        # Compute element rhs vector -> rhs_e
        self._compute_element_rhs(self.rhs_e)

        # Assemble the global rhs vector -> rhs
        self._assemble_rhs(self.rhs_e, self.rhs)
        return self.rhs

    @time_this
    def compute_jacobian(self, rho=1.0):
        """
        Compute the element Jacobian (stiffness) matrices Ke and assemble the
        global K

        Return:
            K: (sparse) global Jacobian matrix
        """
        # Set nodal density
        if not hasattr(rho, "__len__"):
            rho = np.ones(self.nnodes) * rho

        # If performing complex-step verification, cast to complex type
        if rho.dtype == complex:
            self.Ke_mat = self.Ke_mat.astype(complex)

        # Update self.kappa_q
        self._update_material_property(rho)

        # Compute element Jacobian matrix -> Ke_mat
        self._compute_element_jacobian(self.Ke_mat)

        # Assemble global Jacobian -> K
        K = self._assemble_jacobian(self.Ke_mat)
        return K

    @time_this
    def compliance(self, rho, solver="cg"):
        """
        Compute the thermal compliance function given nodal density

        Inputs:
            rho: nodal (usually filtered) density
            solver: cg or direct, note that for a complex step, direct must be
                    used

        Return:
            c: compliance scalar
            u: solution vector
        """
        assert solver == "cg" or solver == "direct"

        # Construct the linear system
        K = self.compute_jacobian(rho)
        rhs = self.compute_rhs()

        # Apply Dirichlet boundary conditions
        K, rhs = self.apply_dirichlet_bcs(K, rhs, enforce_symmetric_K=True)

        # Solve the linear system
        if solver == "direct":
            u = spsolve(K, rhs)
        else:
            ml = pyamg.smoothed_aggregation_solver(K)
            M = ml.aspreconditioner()
            u, fail = cg(K, rhs, tol=1e-8, M=M, atol=0.0)
            if fail:
                raise RuntimeError(f"CG failed with code {fail}")

        c = rhs.dot(u)
        return c, u

    @time_this
    def compliance_grad(self, rho, u):
        """
        Compute the thermal compliance function gradient w.r.t. nodal density
        Inputs:
            rho: nodal (usually filtered) density
            u: solution vector

        Return:
            grad: gradient with respect to rho
        """
        grad = -self._compute_K_dv_sens(rho, u, u)
        return grad

    @time_this
    def volume(self, rho):
        """
        Compute the normalized volume of the design

        Inputs:
            rho: nodal (usually filtered) density
        """
        vol = rho.sum() / self.nnodes
        return vol

    @time_this
    def volume_grad(self, rho):
        """
        Compute the gradient of the normalized volume with respect to rho

        Inputs:
            rho: nodal (usually filtered) density
        """
        grad = np.ones(self.nnodes) / self.nnodes
        return grad

    @time_this
    def _compute_gfun(self, Xq, fun_vals):
        fun_vals[...] = self.gfunc(Xq)
        return

    @time_this
    def _einsum_element_rhs(self, detJq, wq, N, fun_vals, rhs_e):
        rhs_e[...] = np.einsum(
            "ik,k,jk,ik -> ij", detJq, wq, N, fun_vals, optimize=True
        )
        return

    @time_this
    def _compute_element_rhs(self, rhs_e):
        """
        Evaluate element-wise rhs vectors:

            rhs_e = ∑ detJq wq (gN)_q
                    q

        Outputs:
            rhs_e: element-wise rhs, (nelems, nnodes_per_elem * ndof_per_node)
        """
        # Compute shape function and derivatives
        N = self.basis.eval_shape_fun()
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute element mapping -> Xq
        utils.compute_elem_interp(N, self.Xe, self.Xq)

        # Allocate memory for quadrature function values
        if self.fun_vals is None:
            self.fun_vals = np.zeros(self.Xq.shape[0:-1])

        # Evaluate function g at each quadrature point q
        self._compute_gfun(self.Xq, self.fun_vals)

        # Get quadrature weights
        wq = self.quadrature.get_weight()

        # Compute element rhs
        self._einsum_element_rhs(self.detJq, wq, N, self.fun_vals, rhs_e)
        return

    @time_this
    def _einsum_element_jacobian(self, kappa_q, detJq, wq, Ngrad, Ke):
        Ke[...] = np.einsum(
            "iq,iq,q,iqjl,iqkl -> ijk",
            kappa_q,
            detJq,
            wq,
            Ngrad,
            Ngrad,
            optimize=True,
        )
        return

    @time_this
    def _compute_element_jacobian(self, Ke):
        """
        Evaluate element-wise Jacobian matrices

            Ke = ∑ detJq wq ( NxNxT + NyNyT + ...)_q
                 q

        Outputs:
            Ke: element-wise Jacobian matrix, (nelems, nnodes_per_elem *
                ndof_per_node, nnodes_per_elem * ndof_per_node)
        """
        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Get quadrature weights
        wq = self.quadrature.get_weight()

        self._einsum_element_jacobian(self.kappa_q, self.detJq, wq, self.Ngrad, Ke)

        return

    @time_this
    def _einsum_element_sens(self, kappa_q_deriv, detJq, wq, Ngrad, Ke_deriv):
        Ke_deriv[...] = np.einsum(
            "iqo,iq,q,iqjl,iqkl -> ijko",
            kappa_q_deriv,
            detJq,
            wq,
            Ngrad,
            Ngrad,
            optimize=True,
        )
        return

    @time_this
    def _einsum_element_sens_inner(self, conn_dof, phi, psi, Ke_deriv, inner):
        inner[...] = np.einsum(
            "ij,ik,ijko -> io", phi[conn_dof], psi[conn_dof], Ke_deriv
        )
        return

    @time_this
    def _compute_K_dv_sens(self, rho, phi, psi):
        """
        Compute the sensitivity of scalar value (phi^T K psi) w.r.t. nodal
        density rho
        """
        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Get quadrature weights
        wq = self.quadrature.get_weight()

        # Compute material property w.r.t. rho -> self.kappa_q_deriv
        self._update_material_property_deriv(rho)

        self._einsum_element_sens(
            self.kappa_q_deriv, self.detJq, wq, self.Ngrad, self.Ke_deriv
        )

        # Compute the derivative of the inner product for each element
        self._einsum_element_sens_inner(
            self.conn_dof, phi, psi, self.Ke_deriv, self.inner
        )

        # Assemble the derivative
        dfdrho = np.zeros(self.nnodes)
        for i in range(self.nnodes_per_elem):
            np.add.at(dfdrho, self.conn[:, i], self.inner[:, i])
        return dfdrho

    @time_this
    def _update_material_property(self, rho):
        """
        Update material property on quadrature points given nodal density. RAMP
        penalization is used, hence the material kappa is multiplied by a factor
        of rho / (1 + p * (1 - rho))

        Inputs:
            rho: nodal design variable
        """
        # If performing complex-step verification, cast to complex type
        if rho.dtype == complex:
            self.rho_e = self.rho_e.astype(complex)
            self.rho_q = self.rho_q.astype(complex)
            self.kappa_q = self.kappa_q.astype(complex)

        # Compute density at each quadrature point
        utils.scatter_node_to_elem(self.conn, rho, self.rho_e)
        N = self.basis.eval_shape_fun()
        utils.compute_elem_interp(N, self.rho_e, self.rho_q)

        # Compute penalized density at each quadrature point
        self.kappa_q[:] = self.rho_q / (1 + self.p * (1 - self.rho_q))
        return

    @time_this
    def _update_material_property_deriv(self, rho):
        """
        Update material property on quadrature points w.r.t. nodal density. RAMP
        penalization is used, hence the material kappa is multiplied by a factor
        of (1 + p) / (1 + p * (1 - rho))**2

        Inputs:
            rho: nodal design variable
        """
        # If performing complex-step verification, cast to complex type
        if rho.dtype == complex:
            self.rho_e = self.rho_e.astype(complex)
            self.rho_q = self.rho_q.astype(complex)
            self.kappa_q_deriv = self.kappa_q_deriv.astype(complex)

        # Compute density at each quadrature point
        utils.scatter_node_to_elem(self.conn, rho, self.rho_e)
        N = self.basis.eval_shape_fun()
        utils.compute_elem_interp(N, self.rho_e, self.rho_q)

        # Compute penalized density derivative at each quadrature point
        ramp_rho_q_deriv = (1 + self.p) / (1 + self.p * (1 - self.rho_q)) ** 2

        N = self.basis.eval_shape_fun()
        self.kappa_q_deriv[...] = np.einsum("ql,iq -> iql", N, ramp_rho_q_deriv)
        return


class NonlinearPoisson2D(ModelBase):
    """
    The 2-dimensional Nonlinear Poisson equation

    Equation:
    - grad . (h(x)(1.0 + u^2) grad(u))) = g

    Boundary condition:
    u = u0 on ∂Ω

    Weak form:
    ∫ h(x)(1.0 + uq^2) ∇uq ∇vq dΩ = ∫gvq dΩ for test function ve

    Residual:
    R(x, u) = ∫ h(x)(1.0 + uq^2) ue.T B.T B ue - g N dΩ = 0 for test function ve

    where:
        uq = N * ue,
        vq = N * ve,
        ∇uq = ∇N * ue = B * ue,
        ∇vq = ∇N * ve = B * ve,
        B = ∇N = "Ngrad"
    """

    @time_this
    def __init__(
        self,
        X,
        conn,
        dof_fixed,
        dof_fixed_vals,
        quadrature: QuadratureBase,
        basis: BasisBase,
    ):
        ndof_per_node = 1
        super().__init__(
            ndof_per_node, X, conn, dof_fixed, dof_fixed_vals, quadrature, basis
        )
        self.gfun_vals = None
        self.hfun_vals = None
        self.Re = None
        return

    @time_this
    def compute_rhs(self, xdv, u):
        # Compute element rhs vector -> rhs_e
        # self.rhs_e[...] = 0.0
        self._compute_element_rhs(
            xdv,
            u,
            self.rhs_e,
        )

        # Assemble the global rhs vector -> rhs
        self.rhs[...] = 0.0
        self._assemble_rhs(self.rhs_e, self.rhs)
        return self.rhs

    @time_this
    def compute_jacobian(self, xdv, u):
        """
        Compute the element Jacobian (stiffness) matrices Ke and assemble the
        global K

        Return:
            K: (sparse) global Jacobian matrix
        """
        # Compute element Jacobian -> Ke
        self._compute_element_jacobian(xdv, u, self.Ke_mat)

        # Assemble global Jacobian -> K
        K = self._assemble_jacobian(self.Ke_mat)
        return K

    @time_this
    def compute_residual(self, u):
        """
        Compute the element residual vectors and assemble the global residual
        vector

        Inputs:
            u: (ndarray) global solution vector

        Return:
            r: (ndarray) global residual vector
        """
        # Compute element residual -> Re
        self._compute_element_residual(u, self.Re)

        # Assemble the global residual vector -> R
        R = np.zeros(self.nnodes)
        for i in range(self.nnodes_per_elem):
            np.add.at(R, self.conn[:, i], self.Re[:, i])
        return R

    @time_this
    def _compute_gfun(self, Xq, gfun_vals):
        """
        Args:
            Xq: quadrature location in 1st and 2nd global coordinates

        Returns:
            gfun_vals: g function values for each quadrature, (nelems, nquads)
        """
        xvals = Xq[..., 0]
        yvals = Xq[..., 1]
        gfun_vals[...] = (
            1e4
            * xvals
            * (1.0 - xvals)
            * (1.0 - 2.0 * xvals)
            * yvals
            * (1.0 - yvals)
            * (1.0 - 2.0 * yvals)
        )

        return

    @time_this
    def _compute_hfun(self, xdv, Xq, hfun_vals):
        """
        Given the x and y locations return the right-hand-side

        Args:
            xdv: The design variable values
            Xq: quadrature location in 1st and 2nd global coordinates

        Returns:
            hfun_vals: h function values for each quadrature, (nelems, nquads)
        """
        hfun_vals[...] = 0.0
        xvals = Xq[..., 0]
        yvals = Xq[..., 1]
        num_x_vals = np.shape(xdv)[0]
        for k in range(num_x_vals):
            coef = special.binom(num_x_vals - 1, k)
            xarg = coef * (1.0 - xvals) ** (num_x_vals - 1 - k) * xvals**k
            yarg = 4.0 * yvals * (1.0 - yvals)
            hfun_vals[...] += xdv[k] * xarg * yarg
        hfun_vals[...] += 1.0
        return

    @time_this
    def _compute_element_rhs(self, xdv, u, rhs_e):
        """
        Evaluate element-wise rhs vectors:

            rhs_e = ∑ detJq wq (gN)_q

        Args:
            N: shape function values, (nnodes_per_elem, nquads)
            Nderiv: shape function derivatives, (nnodes_per_elem, ndims, nquads)
            detJq: Jacobian determinants, (nelems, nquads)
            Xq: quadrature location in global coordinates, (nelems, nquads, ndims)
            wq: quadrature weights, (nelems, nquads)
            gfun_vals: g function values for each quadrature, (nelems, nquads)
            hfun_vals: h function values for each quadrature, (nelems, nquads)

        Outputs:
            rhs_e: element-wise rhs, (nelems, nnodes_per_elem * nvars_per_node)
        """
        # Compute shape function and derivatives
        N = self.basis.eval_shape_fun()

        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute element mapping -> Xq
        utils.compute_elem_interp(N, self.Xe, self.Xq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Compute element mapping u -> ue -> uq
        ue = np.zeros((self.nelems, self.nnodes_per_elem))
        uq = np.zeros((self.nelems, self.nnodes_per_elem))
        utils.scatter_node_to_elem(self.conn, u, ue)
        utils.compute_elem_interp(N, ue, uq)

        # Allocate memory for quadrature function values
        if self.hfun_vals is None:
            self.hfun_vals = np.zeros(self.Xq.shape[0:-1])

        # Compute g and h function values
        if self.gfun_vals is None:
            self.gfun_vals = np.zeros(self.Xq.shape[0:-1])

        # Evaluate function h
        self._compute_hfun(xdv, self.Xq, self.hfun_vals)
        self._compute_gfun(self.Xq, self.gfun_vals)

        wq = self.quadrature.get_weight()
        rhs_e[...] = np.einsum(
            "nq,nqjl,nqkl,nk -> nj",
            self.detJq * self.hfun_vals * (1.0 + uq**2) * wq,
            self.Ngrad,
            self.Ngrad,
            ue,
        )
        rhs_e[...] -= np.dot(self.detJq * wq * self.gfun_vals, N)

        return

    @time_this
    def _compute_element_jacobian(self, xdv, u, Ke):
        """
        Evaluate element-wise Jacobian matrices:
        Args:
            xdv: The design variable values
            u: The current solution vector

        Outputs:
            Ke: element-wise Jacobian matrix, (nelems, nnodes_per_elem *
                nvars_per_node, nnodes_per_elem * nvars_per_node)

            h(x)(1.0 + uq^2) ∇uq ∇vq = h(x)(1.0 + uq^2) ue.T B.T B ve = Ke ve
            Ke =  h(x)(1.0 + uq^2) B.T B + 2.0 h(x) ue B.T B uq

        where:
            uq = N * ue,
            vq = N * ve,
            ∇uq = ∇N * ue = B * ue,
            ∇vq = ∇N * ve = B * ve,
            B = ∇N = "Ngrad"
        """
        # Compute shape function and derivatives
        N = self.basis.eval_shape_fun()

        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute element mapping -> Xq
        utils.compute_elem_interp(N, self.Xe, self.Xq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Compute element mapping u -> ue -> uq
        ue = np.zeros((self.nelems, self.nnodes_per_elem))
        uq = np.zeros((self.nelems, self.nnodes_per_elem))
        utils.scatter_node_to_elem(self.conn, u, ue)
        utils.compute_elem_interp(N, ue, uq)

        # Allocate memory for quadrature function values
        if self.hfun_vals is None:
            self.hfun_vals = np.zeros(self.Xq.shape[0:-1])

        # Evaluate function h
        self._compute_hfun(xdv, self.Xq, self.hfun_vals)

        wq = self.quadrature.get_weight()
        Ke[...] = np.einsum(
            "nq,q,nqjl,nqkl -> njk",
            self.detJq * self.hfun_vals * (1.0 + uq**2),
            wq,
            self.Ngrad,
            self.Ngrad,
        )
        Ke[...] += np.einsum(
            "nq,nqjl,nqkl,nk,qi -> nji",  # TODO, change ik to ki
            2.0 * self.detJq * self.hfun_vals * uq * wq,
            self.Ngrad,
            self.Ngrad,
            ue,
            N,
        )
        return

    @time_this
    def _compute_element_residual(self, u, Re):
        """
        Evaluate element-wise residual matrices:
        Args:
            xdv: The design variable values
            u: The current solution vector

        Outputs:
            Residual:
            R(x, u) = ∫ h(x)(1.0 + uq^2) ue.T B.T B ue - g N dΩ = 0 for test function ve

            where:
                uq = N * ue,
                vq = N * ve,
                ∇uq = ∇N * ue = B * ue,
                ∇vq = ∇N * ve = B * ve,
                B = ∇N = "Ngrad"
        """
        # Compute shape function and derivatives
        N = self.basis.eval_shape_fun()

        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute element mapping -> Xq
        utils.compute_elem_interp(N, self.Xe, self.Xq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Compute element mapping u -> ue -> uq
        ue = np.zeros((self.nelems, self.nnodes_per_elem))
        uq = np.zeros((self.nelems, self.nnodes_per_elem))
        utils.scatter_node_to_elem(self.conn, u, ue)
        utils.compute_elem_interp(N, ue, uq)

        # Compute g and h function values
        if self.gfun_vals is None:
            self.gfun_vals = np.zeros(self.Xq.shape[0:-1])

        # Evaluate function h
        self._compute_gfun(self.Xq, self.gfun_vals)

        wq = self.quadrature.get_weight()
        Re[...] = np.dot(self.detJq * wq * self.gfun_vals, N)
        return


class LinearElasticity(ModelBase):
    """
    The linear elasticity equation, if is 2-dimensional problem, then plane
    stress condition is assumed

    Integral form using the principle of virtual displacement:
    ∫ eTs dΩ = ∫uTf dΩ

    where:
        e: strain vector
        s: stress vector
        u: nodal displacement
        f: nodal external force
    """

    @time_this
    def __init__(
        self,
        X,
        conn,
        dof_fixed,
        dof_fixed_vals,
        nodal_force,
        quadrature: QuadratureBase,
        basis: BasisBase,
        E=10.0,
        nu=0.3,
        p=0.0,
    ):
        """
        Inputs:
            p: RAMP penalization
        """
        # Infer the problem dimensions
        ndof_per_node = X.shape[1]

        # Call base class's constructor
        super().__init__(
            ndof_per_node, X, conn, dof_fixed, dof_fixed_vals, quadrature, basis
        )

        self.nodal_force = nodal_force

        # Allocate B matrix
        n_stress_tensor = int(ndof_per_node * (ndof_per_node + 1) / 2)
        self.Be = np.zeros(
            (
                self.nelems,
                self.nquads,
                n_stress_tensor,
                self.nnodes_per_elem * ndof_per_node,
            )
        )

        # Allocate material property at quadratures
        self.Cq = np.zeros((self.nelems, self.nquads))
        self.Cqderiv = np.zeros(
            (
                self.nelems,
                self.nquads,
                self.nnodes_per_elem,
            )
        )

        # Allocate rho at element and quadrature
        self.rho_e = np.zeros((self.nelems, self.nnodes_per_elem))
        self.rho_q = np.zeros((self.nelems, self.nquads))

        self.inner = np.zeros((self.nelems, self.nnodes_per_elem))
        self.Ke_deriv = np.zeros(
            (
                self.nelems,
                self.nnodes_per_elem * self.ndof_per_node,
                self.nnodes_per_elem * self.ndof_per_node,
                self.nnodes_per_elem,
            )
        )
        self.p = p

        # Compute constitutive matrix
        if ndof_per_node == 2:
            self.C0 = E * np.array(
                [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
            )
            self.C0 *= 1.0 / (1.0 - nu**2)
        else:
            self.C0 = np.zeros((6, 6))
            self.C0[0, 0] = self.C0[1, 1] = self.C0[2, 2] = 1 - nu
            self.C0[0, 1] = self.C0[0, 2] = self.C0[1, 0] = nu
            self.C0[1, 2] = self.C0[2, 0] = self.C0[2, 1] = nu
            self.C0[3, 3] = self.C0[4, 4] = self.C0[5, 5] = 0.5 - nu
            self.C0 *= E / ((1 + nu) * (1 - 2 * nu))
        return

    @time_this
    def compute_rhs(self):
        """
        It turns out for this problem, the rhs (before applying bcs) is just nodal force
        """
        self.rhs[
            self.dof_each_node[list(self.nodal_force.keys())].flatten()
        ] = np.array(list(self.nodal_force.values())).flatten()
        return self.rhs

    @time_this
    def compute_jacobian(self, rho=1.0):
        """
        Compute the element Jacobian (stiffness) matrices Ke and assemble the
        global K

        Return:
            K: (sparse) global Jacobian matrix
        """
        # Set nodal density
        if not hasattr(rho, "__len__"):
            rho = np.ones(self.nnodes) * rho

        # If performing complex-step verification, cast to complex type
        if rho.dtype == complex:
            self.Ke_mat = self.Ke_mat.astype(complex)

        # Update self.Cq
        self._update_material_property(rho)

        # Compute element Jacobian matrix -> Ke
        self._compute_element_jacobian(self.Ke_mat)

        # Assemble global Jacobian -> K
        K = self._assemble_jacobian(self.Ke_mat)
        return K

    @time_this
    def compliance(self, rho, solver="cg"):
        """
        Compute the compliance function given nodal density

        Inputs:
            rho: nodal (usually filtered) density
            solver: cg or direct, note that for a complex step, direct must be
                    used

        Return:
            c: compliance scalar
            u: solution vector
        """
        assert solver == "cg" or solver == "direct"

        # Construct the linear system
        K = self.compute_jacobian(rho)
        rhs = self.compute_rhs()

        # Apply Dirichlet boundary conditions
        K, rhs = self.apply_dirichlet_bcs(K, rhs, enforce_symmetric_K=True)

        # Solve the linear system
        if solver == "direct":
            u = spsolve(K, rhs)
        else:
            ml = pyamg.smoothed_aggregation_solver(K)
            M = ml.aspreconditioner()
            u, fail = cg(K, rhs, tol=1e-8, M=M, atol=0.0)
            if fail:
                raise RuntimeError(f"CG failed with code {fail}")

        c = rhs.dot(u)
        return c, u

    @time_this
    def compliance_grad(self, rho, u):
        """
        Compute the compliance function gradient w.r.t. nodal density
        Inputs:
            rho: nodal (usually filtered) density
            u: solution vector

        Return:
            grad: gradient with respect to rho
        """
        grad = -self._compute_K_dv_sens(rho, u, u)
        return grad

    @time_this
    def volume(self, rho):
        """
        Compute the normalized volume of the design

        Inputs:
            rho: nodal (usually filtered) density
        """
        vol = rho.sum() / self.nnodes
        return vol

    @time_this
    def volume_grad(self, rho):
        """
        Compute the gradient of the normalized volume with respect to rho

        Inputs:
            rho: nodal (usually filtered) density
        """
        grad = np.ones(self.nnodes) / self.nnodes
        return grad

    @time_this
    def _compute_K_dv_sens(self, rho, phi, psi):
        """
        Compute the sensitivity of scalar value (phi^T K psi) w.r.t. nodal
        density rho
        """
        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Compute Be matrix -> Be
        self._compute_element_Bmat(self.Ngrad, self.Be)

        # Get quadrature weights
        wq = self.quadrature.get_weight()

        # Compute material property w.r.t. rho -> self.Cqderiv
        self._update_material_property_deriv(rho)

        # Compute derivative of Ke w.r.t. rho
        self.Ke_deriv[...] = np.einsum(
            "iq,q,iqnj,iqo,nm,iqmk->ijko",
            self.detJq,
            wq,
            self.Be,
            self.Cqderiv,
            self.C0,
            self.Be,
            optimize=True,
        )

        # Compute the derivative of the inner product for each element
        self.inner[...] = np.einsum(
            "ij,ik,ijko -> io", phi[self.conn_dof], psi[self.conn_dof], self.Ke_deriv
        )

        # Assemble the derivative
        dfdrho = np.zeros(self.nnodes)
        for i in range(self.nnodes_per_elem):
            np.add.at(dfdrho, self.conn[:, i], self.inner[:, i])
        return dfdrho

    @time_this
    def _update_material_property(self, rho):
        """
        Update material property on quadrature points given nodal density. RAMP
        penalization is used, hence the material C is multiplied by a factor of
        rho / (1 + p * (1 - rho))

        Inputs:
            rho: nodal design variable
        """
        # If performing complex-step verification, cast to complex type
        if rho.dtype == complex:
            self.rho_e = self.rho_e.astype(complex)
            self.rho_q = self.rho_q.astype(complex)
            self.Cq = self.Cq.astype(complex)

        # Compute density at each quadrature point
        utils.scatter_node_to_elem(self.conn, rho, self.rho_e)
        N = self.basis.eval_shape_fun()
        utils.compute_elem_interp(N, self.rho_e, self.rho_q)

        # Compute penalized density at each quadrature point
        self.Cq[:] = self.rho_q / (1 + self.p * (1 - self.rho_q))
        return

    @time_this
    def _update_material_property_deriv(self, rho):
        """
        Update material property on quadrature points w.r.t. nodal density. RAMP
        penalization is used, hence the material C0 is multiplied by a factor of
        (1 + p) / (1 + p * (1 - rho))**2

        Inputs:
            rho: nodal design variable
        """
        # If performing complex-step verification, cast to complex type
        if rho.dtype == complex:
            self.rho_e = self.rho_e.astype(complex)
            self.rho_q = self.rho_q.astype(complex)
            self.Cqderiv = self.Cqderiv.astype(complex)

        # Compute density at each quadrature point
        utils.scatter_node_to_elem(self.conn, rho, self.rho_e)
        N = self.basis.eval_shape_fun()
        utils.compute_elem_interp(N, self.rho_e, self.rho_q)

        # Compute penalized density derivative at each quadrature point
        ramp_rho_q_deriv = (1 + self.p) / (1 + self.p * (1 - self.rho_q)) ** 2

        N = self.basis.eval_shape_fun()
        self.Cqderiv[...] = np.einsum("ql,iq -> iql", N, ramp_rho_q_deriv)
        return

    @time_this
    def _compute_element_Bmat(self, Ngrad, Be):
        """
        Compute element B matrix

        Input:
            Ngrad: shape function gradient, (nelems, nquads, nnodes_per_elem,
                   ndims)

        Output:
            Be: function that maps u to strain, (nelems, nquads, ndof_per_node,
            nnodes_per_elem * ndof_per_node)
        """
        if self.ndims == 2:
            Nx = Ngrad[:, :, :, 0]
            Ny = Ngrad[:, :, :, 1]
            Be[:, :, 0, ::2] = Nx
            Be[:, :, 1, 1::2] = Ny
            Be[:, :, 2, ::2] = Ny
            Be[:, :, 2, 1::2] = Nx

        elif self.ndims == 3:
            Nx = Ngrad[:, :, :, 0]
            Ny = Ngrad[:, :, :, 1]
            Nz = Ngrad[:, :, :, 2]
            Be[:, :, 0, 0::3] = Nx
            Be[:, :, 1, 1::3] = Ny
            Be[:, :, 2, 2::3] = Nz

            Be[:, :, 3, 0::3] = Ny
            Be[:, :, 3, 1::3] = Nx

            Be[:, :, 4, 1::3] = Nz
            Be[:, :, 4, 2::3] = Ny

            Be[:, :, 5, 0::3] = Nz
            Be[:, :, 5, 2::3] = Nx

        return

    @time_this
    def _einsum_element_jacobian(self, detJq, wq, Be, Cq, C0, Ke):
        Ke[...] = np.einsum(
            "iq,q,iqnj,iq,nm,iqmk->ijk",
            detJq,
            wq,
            Be,
            Cq,
            C0,
            Be,
            optimize=True,
        )
        return

    @time_this
    def _compute_element_jacobian(self, Ke):
        """
        Evaluate element-wise stiffness matrix Ke

        ∫ eTs dΩ can be integrated numerically using quadrature:

            ∫eTs dΩ ~= ∑ detJq wq eTs
                       q

        where e = Bu, s = Ce = CBu.  Then, Ke can be computed by taking the 2nd
        order derivatives w.r.t. u:

            Ke = ∑ detJq wq (BTCB)q
                  q

        Outputs:
            Ke: element-wise Jacobian matrix, (nelems, nnodes_per_elem *
                ndof_per_node, nnodes_per_elem * ndof_per_node)
        """
        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Compute Be matrix -> Be
        self._compute_element_Bmat(self.Ngrad, self.Be)

        # Get quadrature weights
        wq = self.quadrature.get_weight()

        self._einsum_element_jacobian(self.detJq, wq, self.Be, self.Cq, self.C0, Ke)
        return


class Helmholtz(ModelBase):
    """
    Solve the 2- or 3-dimensional Helmholtz equation:
    -r^2 ∆rho + rho = x

    with (usually naturally satisfied) Neumann boundary condition:
    drho/dn = 0

    Helmholtz equation can be used a the filter method for topology optimization
    applications. In this case, x is the (nodal) raw design variable, and rho
    is the filtered density field, with r controlling the ``filter radius''
    """

    @time_this
    def __init__(self, r0, X, conn, quadrature: QuadratureBase, basis: BasisBase):
        """
        Inputs:
            r0: filter radius
        """
        super().__init__(1, X, conn, [], None, quadrature, basis)

        self.r0 = r0
        self.Re = np.zeros((self.nelems, self.nnodes_per_elem, self.nnodes_per_elem))
        self._compute_element_jacobian_and_rhs(self.Ke_mat, self.Re)
        self.R = self._assemble_jacobian(self.Re)
        self.RT = self.R.transpose()
        self.K = self._assemble_jacobian(self.Ke_mat)
        # self.Ksolve = sparse.linalg.factorized(self.K.tocsc())
        self.Ksolve = pyamg.ruge_stuben_solver(self.K)
        return

    @time_this
    def apply(self, x):
        """
        Apply the Helmholtz filter to x -> rho
        """
        rho = self.Ksolve.solve(self.compute_rhs(x), tol=1e-8)
        return rho

    @time_this
    def apply_gradient(self, gradrho):
        """
        Apply the Helmholtz filter to the gradient ∇_rho -> ∇x
        """
        gradx = self.RT.dot(self.Ksolve.solve(gradrho, tol=1e-8))
        return gradx

    @time_this
    def compute_rhs(self, x):
        self.rhs[:] = self.R.dot(x)
        return self.rhs

    @time_this
    def compute_jacobian(self):
        return self.K

    @time_this
    def _einsum_element_jacobian(self, detJq, r0, wq, Ngrad, Ke):
        Ke[...] = np.einsum(
            "iq,q,iqjl,iqkl -> ijk", detJq * r0**2, wq, Ngrad, Ngrad, optimize=True
        )
        return

    @time_this
    def _einsum_element_rhs(self, detJq, wq, N, Re):
        Re[...] = np.einsum("iq,q,qj,qk -> ijk", detJq, wq, N, N, optimize=True)
        return

    @time_this
    def _compute_element_jacobian_and_rhs(self, Ke, Re):
        """
        Evaluate element-wise Jacobian matrices and rhs, where


            Ke = ∑ detJq wq [r**2 (NxNxT + NyNyT) + NNT]_q
                 q

            Re = [∑ detJq wq (NNT)_q]
                  q

        Outputs:
            Ke: element-wise Jacobian matrix, (nelems, nnodes_per_elem *
                ndof_per_node, nnodes_per_elem * ndof_per_node)
            Re: element-wise matrix for rhs, (nelems, nnodes_per_elem *
                ndof_per_node, nnodes_per_elem * ndof_per_node)
        """
        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Get quadrature weights
        wq = self.quadrature.get_weight()

        # Get shape functions
        N = self.basis.eval_shape_fun()

        self._einsum_element_rhs(self.detJq, wq, N, Re)
        self._einsum_element_jacobian(self.detJq, self.r0, wq, self.Ngrad, Ke)
        Ke[...] += Re[...]
        return


class A2DWrapper(ModelBase):
    """
    An experimental class that wraps around a2d (https://github.com/gjkennedy/a2d)
    """

    @time_this
    def __init__(self, X, conn, dof_fixed, dof_fixed_vals, a2d, problem_info):
        """
        Inputs:
            ...
            a2d: the a2d library
            problem_info: the problem information, dictionary
                          problem_info = {
                              'type': 'elasticity',
                              'E': E,
                              'nu': nu
                          }
                          or
                          problem_info = {
                              'type': 'helmholtz',
                              'r0': r0
                          }

        """

        nelems = conn.shape[0]
        nnodes = X.shape[0]

        if problem_info["type"] == "elasticity":
            a2dmodel = a2d.Elasticity(nelems, nnodes)
        elif problem_info["type"] == "helmholtz":
            a2dmodel = a2d.Helmholtz(nelems, nnodes)
        else:
            raise ValueError(f"Unknown problem_info {problem_info}")

        # Set connectivity numpy array and update underlying c++ data
        self.conn = np.array(a2dmodel.get_conn(), copy=False)
        self.conn[:] = conn[:]

        # Set nodal location numpy array and update underlying c++ data
        self.X = np.array(a2dmodel.get_nodes(), copy=False)
        self.X[:] = X[:]
        a2dmodel.reset_nodes()  # Required after node location change

        # Set material property numpy array and update underlying c++ data
        self.data = np.array(a2dmodel.get_quad_data(), copy=False)
        if problem_info["type"] == "elasticity":
            E = problem_info["E"]
            nu = problem_info["nu"]
            mu = E / (2 * (1 + nu))
            lam = E * nu / (1 + nu) / (1 - 2 * nu)
            self.data[:, :, 0] = mu  # Lame parameter: mu
            self.data[:, :, 1] = lam  # Lame parameter: lambda
        elif problem_info["type"] == "helmholtz":
            self.data[:] = problem_info["r0"]

        # Set solution numpy array and update underlying c++ data
        self.U = np.array(a2dmodel.get_solution(), copy=False)
        self.U[:] = 0.0
        a2dmodel.reset_solution()  # Required after solution change

        # Set element-wise Jacobian numpy array and update underlying c++ data
        self.jac = np.array(a2dmodel.get_elem_jac(), copy=False)
        self.jac[:] = 0.0

        self.a2dmodel = a2dmodel

        # Call constructor for the parent class, note that for now only 8-node
        # block element is supported
        ndof_per_node = self.U.shape[-1]
        quadrature = QuadratureBlock3D()
        basis = BasisBlock3D(quadrature)
        super().__init__(
            ndof_per_node, X, conn, dof_fixed, dof_fixed_vals, quadrature, basis
        )

    @time_this
    def compute_rhs(self):
        return None  # TODO

    @time_this
    def compute_jacobian(self):
        self._compute_jacobian_tensor()
        self._jacobian_tensor_to_mat(self.jac, self.Ke_mat)
        K = self._assemble_jacobian(self.Ke_mat)
        return K

    @time_this
    def _compute_jacobian_tensor(self):
        self.a2dmodel.add_jacobians()
        return


class Assembler:
    @time_this
    def __init__(self, model: ModelBase):
        """
        The finite element problem assembler.

        Args:
            model: the problem model instance
        """
        self.model = model
        return

    @time_this
    def solve(self, method="gmres"):
        """
        Perform the static analysis
        """
        # Construct the linear system
        K = self.model.compute_jacobian()
        rhs = self.model.compute_rhs()

        # Apply Dirichlet boundary conditions
        K, rhs = self.model.apply_dirichlet_bcs(K, rhs, enforce_symmetric_K=True)

        # Solve the linear system
        u = self._solve_linear_system(K, rhs, method)
        return u

    @time_this
    def solve_nonlinear(
        self, method="gmres", xdv=None, u0=None, tol=1e-10, atol=1e-12, max_iter=10
    ):
        """
        Perform the static analysis
        """
        # Set the initial guess as u = 0
        if u0 is None:
            u = np.zeros(self.model.nnodes)
        else:
            u = u0

        for k in range(max_iter):
            # Construct the linear system
            K = self.model.compute_jacobian(xdv, u)
            res = self.model.compute_rhs(xdv, u)

            # Apply Dirichlet boundary conditions
            self.model.apply_dirichlet_bcs(K, res, enforce_symmetric_K=False)
            res_norm = np.sqrt(np.dot(res, res))
            print("pyfem", "{0:5d} {1:25.15e}".format(k, res_norm))

            if k == 0:
                res_norm_init = res_norm
            elif res_norm < tol * res_norm_init or res_norm < atol:
                break

            update = self._solve_linear_system(K, res, method)
            u -= update

        return u

    @time_this
    def plot(self, u, ax, **kwargs):
        """
        Create a 2-dimensional contour plot for a scalar variable u
        """
        nelems = self.model.nelems
        conn = self.model.conn
        X = self.model.X
        nnodes_per_elem = self.model.nnodes_per_elem

        # Create the triangles
        if nnodes_per_elem == 4:
            triangles = np.zeros((2 * nelems, 3), dtype=int)
            triangles[:nelems, 0] = conn[:, 0]
            triangles[:nelems, 1] = conn[:, 1]
            triangles[:nelems, 2] = conn[:, 2]

            triangles[nelems:, 0] = conn[:, 0]
            triangles[nelems:, 1] = conn[:, 2]
            triangles[nelems:, 2] = conn[:, 3]
        elif nnodes_per_elem == 3:
            triangles = conn
        else:
            raise ValueError("unsupported element type")

        # Create the triangulation object
        tri_obj = tri.Triangulation(X[:, 0], X[:, 1], triangles)

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)
        return

    @time_this
    def _solve_linear_system(self, K, rhs, method):
        """
        Solve the linear system

        Args:
            method: direct or gmres

        Return:
            u: solution
        """
        if method == "direct":
            u = spsolve(K, rhs)
        else:
            ml = pyamg.smoothed_aggregation_solver(K)
            M = ml.aspreconditioner()
            u, fail = gmres(K, rhs, M=M)
            if fail:
                raise RuntimeError(f"GMRES failed with code {fail}")
        return u


class ProblemCreator:
    """
    Utility to create problem mesh, boundary condition and load
    """

    @time_this
    def __init__(self, nnodes_x, nnodes_y, nnodes_z=None, element_type="quad"):
        """
        Create a 2d problem instance if nnodes_z is None, otherwise create a
        3d problem instance.

        Inputs:
            nnodes_x, nnodes_y, nnodes_z: number of nodes in x, y, (z) directions
            element_type: type of the finite element, currently the following
                          types are supported:
                              - tri: 2d 3-node triangle
                              - quad: 2d 4-node quadrilateral
                              - block: 3d 8-node hexahedron
        """
        # Set problem dimension and check inputs
        if nnodes_z is None:
            self.ndims = 2
            nnodes_z = 1
            assert element_type == "quad" or element_type == "tri"
        else:
            self.ndims = 3
            assert element_type == "block"

        nnodes = nnodes_x * nnodes_y * nnodes_z
        Lx = (nnodes_x - 1) / (nnodes_y - 1)
        Ly = 1.0
        Lz = (nnodes_z - 1) / (nnodes_y - 1)
        x = np.linspace(0, Lx, nnodes_x)
        y = np.linspace(0, Ly, nnodes_y)
        z = np.linspace(0, Lz, nnodes_z)
        nodes3d = np.arange(0, nnodes_x * nnodes_y * nnodes_z).reshape(
            (nnodes_z, nnodes_y, nnodes_x)
        )
        X = np.zeros((nnodes, 3))
        for k in range(nnodes_z):
            for j in range(nnodes_y):
                for i in range(nnodes_x):
                    X[i + j * nnodes_x + k * nnodes_x * nnodes_y, :] = x[i], y[j], z[k]

        temp_nelems_x = nnodes_x - 1
        temp_nelems_y = nnodes_y - 1
        temp_nelems_z = nnodes_z - 1

        # Set the connectivity
        if element_type == "quad":
            nelems = temp_nelems_x * temp_nelems_y
            nnodes_per_elem = 4
            conn = np.zeros((nelems, nnodes_per_elem), dtype=int)
            for j in range(temp_nelems_y):
                for i in range(temp_nelems_x):
                    conn[i + j * temp_nelems_x, 0] = nodes3d[0, j, i]
                    conn[i + j * temp_nelems_x, 1] = nodes3d[0, j, i + 1]
                    conn[i + j * temp_nelems_x, 2] = nodes3d[0, j + 1, i + 1]
                    conn[i + j * temp_nelems_x, 3] = nodes3d[0, j + 1, i]

        elif element_type == "tri":
            nelems = temp_nelems_x * temp_nelems_y * 2
            nnodes_per_elem = 3
            conn = np.zeros((nelems, nnodes_per_elem), dtype=int)
            for j in range(temp_nelems_y):
                for i in range(temp_nelems_x):
                    quad_idx = i + j * temp_nelems_x
                    conn[2 * quad_idx, 0] = nodes3d[0, j, i]
                    conn[2 * quad_idx, 1] = nodes3d[0, j, i + 1]
                    conn[2 * quad_idx, 2] = nodes3d[0, j + 1, i + 1]

                    conn[2 * quad_idx + 1, 0] = nodes3d[0, j + 1, i + 1]
                    conn[2 * quad_idx + 1, 1] = nodes3d[0, j + 1, i]
                    conn[2 * quad_idx + 1, 2] = nodes3d[0, j, i]

        elif element_type == "block":
            nelems = temp_nelems_x * temp_nelems_y * temp_nelems_z
            nnodes_per_elem = 8
            conn = np.zeros((nelems, nnodes_per_elem), dtype=int)
            for k in range(temp_nelems_z):
                for j in range(temp_nelems_y):
                    for i in range(temp_nelems_x):
                        # fmt: off
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 0] = nodes3d[k, j, i]
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 1] = nodes3d[k, j, i + 1]
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 2] = nodes3d[k, j + 1, i + 1]
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 3] = nodes3d[k, j + 1, i]
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 4] = nodes3d[k + 1, j, i]
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 5] = nodes3d[k + 1, j, i + 1]
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 6] = nodes3d[k + 1, j + 1, i + 1]
                        conn[i + j * temp_nelems_x + k * temp_nelems_x * temp_nelems_y, 7] = nodes3d[k + 1, j + 1, i]
                        # fmt: on

        else:
            raise ValueError(f"unknown element_type: {element_type}")

        self.nnodes_x = nnodes_x
        self.nnodes_y = nnodes_y
        self.nnodes_z = nnodes_z
        self.nnodes = nnodes_x * nnodes_y * nnodes_z
        self.nodes3d = nodes3d
        self.conn = conn
        self.X = X[:, 0 : self.ndims]

        return

    @time_this
    def create_poisson_problem(self):
        # Set fixed dof
        dof_fixed = []
        for k in range(self.nnodes_z):
            for j in range(self.nnodes_y):
                dof_fixed.append(self.nodes3d[k, j, 0])
        return self.conn, self.X, dof_fixed

    @time_this
    def create_linear_elasticity_problem(self):
        # Set fixed dof
        dof_fixed = []
        for k in range(self.nnodes_z):
            for j in range(self.nnodes_y):
                for n in range(self.ndims):
                    dof_fixed.append(self.ndims * self.nodes3d[k, j, 0] + n)

        # Set nodal force
        nodal_force = {}
        for k in range(self.nnodes_z):
            # for j in range(self.nnodes_y):
            #     if j < self.nnodes_y / 10:
            #         nodal_force[self.nodes3d[k, j, -1]] = [0.0, -1.0, 0.0][
            #             0 : self.ndims
            #         ]
            nodal_force[self.nodes3d[k, 0, -1]] = [0.0, -1.0, 0.0][0 : self.ndims]

        return self.conn, self.X, dof_fixed, nodal_force

    @time_this
    def create_helmhotz_problem(self):
        x = np.zeros(self.nnodes)
        idx = 0
        for k in range(self.nnodes_z):
            for j in range(self.nnodes_y):
                for i in range(self.nnodes_x):
                    if (
                        i < self.nnodes_x / 2
                        and j < self.nnodes_y / 2
                        and k < self.nnodes_z / 2
                    ):
                        x[idx] = 0.95
                    else:
                        x[idx] = 1e-3
                    idx += 1
        return self.conn, self.X, x
