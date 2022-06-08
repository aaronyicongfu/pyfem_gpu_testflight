import sys

sys.path.append("../..")
from parse_inp import InpParser


inp_file = "sampleFile.inp"
parser = InpParser(inp_file)

# Parse the inp
conn, X, groups = parser.parse()

# Export data to vtk
parser.to_vtk()
