### Import the functions dir

# import GarnetDiffusion.functions

import GarnetDiffusion.functions.bulk_rock_functions
import GarnetDiffusion.functions.garnet_growth
import GarnetDiffusion.functions.MAGEMin_functions



### Fix statically linked python
from julia.api import Julia
jl = Julia(compiled_modules=False)

