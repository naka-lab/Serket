# Symbol Emergence in Robotics tool KIT (SERKET)

This is an implementation of SERKET proposed in the following paper. 

Tomoaki Nakamura, Takayuki Nagai and Tadahiro Taniguchi, "SERKET: An Architecture for Connecting Stochastic Models to Realize a Large-Scale Cognitive Model", Frontiers in Neurorobotics, vol. 12, article 25, pp. 1-16, Jun. 2018 [(PDF)](https://www.frontiersin.org/articles/10.3389/fnbot.2018.00025/full)


## Instration

Clone the repository fromt the GitHub: 

```
git clone https://github.com/naka-tomo/serket.git
```

Install dependency packages (You can skip if you have already installed these packages. ):

```
pip install numpy
pip install opencv-python
```

## Simple Example of Serket

First, the toy dataset of six observations are generated. Here, these are assumed to be generated from three categories, which are latent variables.
```
data = [
    [10, 2, 1], 
    [8, 1, 1],
    [2, 7, 0],
    [1, 11, 2],
    [1, 1, 8],
    [2, 1, 1]
]

data_category = [0, 0, 1, 1, 2, 2 ]
```

Then, we define the modules. `srk.Observation` is a module which send the observation to another module, and `mlda.MLDA` (multimodal latent Dirichlet allocation) is a modules for an unsupervised classification. Here, we define MLDA that classifies dataset into three classes. 
By using the optional argment `category`, the classification accuracy can be computed automatically. 

```
obs = srk.Observation( data )
mlda1 = mlda.MLDA( 3 , category=data_category )
```

By connecting modules, the model is constructed and the parameters of the model are estimated by `update` method.

```
mlda1.connect( obs )
mlda1.update()
```

If you success the training the model, you can find `module000_mlda` directory in your working directory, which contains the results of the training. 
Classification accuracy that is computed automatically is included in `module000_mlda/000/acc.txt`. It might be 1.00, which represents the all observation can be classified correctly. 

