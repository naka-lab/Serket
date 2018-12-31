---
layout: default
---
# Installation

Clone the repository from GitHub:

```
git clone https://github.com/naka-tomo/serket.git
```

Install dependency packages (you can skip this step if you have already installed these packages):

```
pip install numpy
pip install opencv-python
pip install numba
pip install scipy
pip install tensorflow
pip install librosa==0.5.1
```

# Simple Example of Serket

First, the toy dataset of six observations is generated.
Here, these are assumed to be generated from three categories of latent variables.
```
data = [
    [10, 2, 1],
    [8, 1, 1],
    [2, 7, 0],
    [1, 11, 2],
    [1, 1, 8],
    [2, 1, 1]
]

data_category = [0, 0, 1, 1, 2, 2]
```

Next, we define the modules.
`srk.Observation` is a module that sends observations to another module and `mlda.MLDA` (multimodal latent Dirichlet allocation) is a module for unsupervised classification.
Here, we define a kind of MLDA that classifies the dataset into three classes.
By using the optional argument `category`, the classification accuracy can be computed automatically.

```
obs = srk.Observation( data )
mlda1 = mlda.MLDA( 3 , category=data_category )
```

By connecting modules, the model is constructed and the parameters of the model are estimated by the `update` method.

```
mlda1.connect( obs )
mlda1.update()
```

If the model was successfully trained, you can find the `module000_mlda` directory in your working directory, which contains the training results.
The classification accuracy that is computed automatically is saved in `module000_mlda/000/acc.txt`.
It might indicate a value of 1.00, which represents the all observations were classified correctly.
