import os
import numpy as np
import argparse

class InpParser:
    """
    Parse the Abaqus input file .inp
    """
    def __init__(self, inp_name):
        self.inp_name = inp_name
        self.X = list()
        self.conn = dict()
        self.nnodes = None
        self.nelems = None
        return

    def _line_to_list(self, line, dtype=float, drop_index=False, offset=0):
        """
        Convert a line of string to list of numbers with mixed types

        Inputs:
            line: a line of text file
            dtype: python built-in data type
            drop_index: if true, discard the first number in line

        Return:
            vals: list of values, if fail, return None
        """
        vals = [val.strip() for val in line.strip().split(',')]
        if not vals[0].isnumeric():
            return None

        vals = [dtype(v) + offset for v in vals]
        if drop_index:
            vals = vals[1:]
        return vals


    def _parse_nodes(self, fh, X):
        """
        Get nodal information from node section. Node section is the lines
        after *Node.
        Note: We assume that nodes are numbered as 1, 2, 3, ...

        Inputs:
            fh: inp file handle

        Outputs:
            X: list of xyz coordinates

        Return:
            line: the next line of data chunk, if None then EOF is hit
        """
        next_line = None

        # Get nodal numbering and xyz coordinates
        for line in fh:
            xyz = self._line_to_list(line, dtype=float, drop_index=True)

            # Break when data chunk ends
            if not xyz:
                next_line = line
                break

            # Add node to the dictionary
            X.append(xyz)

        return next_line

    def _parse_elems(self, fh, etype, conn):
        """
        Get element mapping. Element mapping data chunk is after *Element.
        Note: We assume that elements are numbered as 1, 2, 3, ...

        Inputs:
            fh: inp file handle
            etype: element type

        Outputs:
            conn: dictionary of connectivities, conn[etype] = list of conns

        Return:
            line: the next line of data chunk, if None then EOF is hit
        """
        next_line = None

        if etype not in conn.keys():
            conn[etype] = []

        # Get element numbering and nodal indices
        for line in fh:
            # Note that offset = 1 because Abaqus uses 1-based indexing, while 0-based
            # indexing is what we want
            _conn = self._line_to_list(line, dtype=int, drop_index=True, offset=-1)

            # Break when data chunk ends
            if not _conn:
                next_line = line
                break

            conn[etype].append(_conn)
        return next_line

    def parse_inp(self):
        """
        Parse the inp file.
        """
        # Load data from inp
        with open(self.inp_name) as fh:
            # Move to node section
            for line in fh:
                if line.strip() == '*Node':
                    break

            # Get nodes
            self._parse_nodes(fh, self.X)

            # Get element connectivity
            self._parse_elems(fh, 'C3D8R', self.conn)
            self._parse_elems(fh, 'C3D10', self.conn)

        # Convert to numpy array
        self.X = np.array(self.X)
        self.conn = {key: np.array(val) for key, val in self.conn.items()}

        # Count nodes and elements
        self.nnodes = self.X.shape[0]
        self.nelems = np.sum([len(v) for k, v in self.conn.items()])

        return

    def to_vtk(self, vtk_name=None):
        """
        Generate a vtk from inp.
        """
        if vtk_name is None:
            vtk_name = '{:s}.vtk'.format(os.path.splitext(self.inp_name)[0])

        print(f'Converting {self.inp_name} to {vtk_name}...', end='')

        # Create a empty vtk file and write headers
        with open(vtk_name, 'w') as fh:
            fh.write('# vtk DataFile Version 3.0\n')
            fh.write('my example\n')
            fh.write('ASCII\n')
            fh.write('DATASET UNSTRUCTURED_GRID\n')

            # Write nodal points
            fh.write('POINTS {:d} double\n'.format(self.nnodes))
            for i in range(self.nnodes):
                fh.write('{:f} {:f} {:f}\n'.format(self.X[i, 0], self.X[i, 1],
                    self.X[i, 2]))

            einfo = {'C3D8R': {'nnode': 8, 'vtk_type': 12},
                            'C3D10': {'nnode':10, 'vtk_type': 24}}

            # Write connectivity
            N = self.nelems
            size = np.sum([len(v)*(1 + einfo[k]['nnode'])
                for k, v in self.conn.items()])
            fh.write(f'CELLS {N} {size}\n')
            for etype, conn in self.conn.items():
                for c in conn:
                    node_idx = f'{c}'
                    node_idx = node_idx[1:-1]  # remove square bracket [ and ]
                    npts = einfo[etype]['nnode']
                    fh.write(f'{npts} {node_idx}\n')

            # Write cell type
            fh.write(f'CELL_TYPES {self.nelems}\n')
            for etype, conn in self.conn.items():
                for c in conn:
                    vtk_type = einfo[etype]['vtk_type']
                    fh.write(f'{vtk_type}\n')

        print('done')
        return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inp', type=str, metavar='[inp file]')
    args = p.parse_args()
    inp_parser = InpParser(args.inp)
    inp_parser.parse_inp()
    inp_parser.to_vtk()
