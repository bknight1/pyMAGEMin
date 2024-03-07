### Import the functions dir

# import GarnetDiffusion.functions

import GarnetDiffusion.functions.bulk_rock_functions
import GarnetDiffusion.functions.garnet_growth
import GarnetDiffusion.functions.MAGEMin_functions


### 
from julia.api import Julia
jl = Julia(compiled_modules=False)

'''
fix for:
Your Python interpreter "../python3"
is statically linked to libpython.  
Currently, PyJulia does not fully
support such Python interpreter.
'''