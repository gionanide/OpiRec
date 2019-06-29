Python scripts shared folder to initialize some global properties.

Python scripts shared folder to initialize some global properties.

## Cuda properties


```python
#-------------------------------------------------------------> Tensorflow Session properties

import gpu_initializations as gpu_init

#and call the function as follows
core = #choose between 'GPU'/'CPU'
memory = #choose between 'dynamically'/'fractions'
parallel = True/False

sess = gpu_init.CUDA_init(core=core,memory=memory,parallel=parallel)

