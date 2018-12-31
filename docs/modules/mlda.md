---
layout: default
---
## Multimodal Latent Dirichlet Allocation (MLDA)

```
mlda.MLDA( K, weights=None, itr=100, name="mlda", category=None, mode="learn" )
```

`mlda.MLDA` is a module for unsupervised classification based on multimodal latent Dirichlet allocation.
It computes the probabilities that each data element is classified into each class.
Modal features of the data are generated based on the classification.
The probabilities and generated features are sent to the connected modules.


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| K         | int | Number of clusters |
| weights  | array | Weight for each modality |
| itr       | int | Number of iterations |
| name      | string | Module name |
| category  | array | Correct class labels |
| mode      | string | Choose from learning mode ("learn") or recognition mode ("recog") |


### Methods

- .connect()  
This method connects this module to observations or modules and constructs the model.
- .update()  
This method estimates model parameters and computes probabilities.
The module estimates model parameters in "learn" mode and predicts classes of novel data in "recog" mode.
If training is successful, then the `module{n}_mlda` directory is created.
The following files are saved in the directory ({mode} contains the selected mode (learn or recog)):
    - `model.pickle`: The model parameters.
    - `acc_{mode}.txt`: The accuracy calculated if the optional argument `category` is set.
    - `categories_{mode}.txt`: The classes into which each data element is classified.
    - `Pdz_{mode}.txt`: The probabilities that each data element is classified into a class.
    - `Pmdw[i]_{mode}.txt`: The probabilities that modal features of the data are generated based on the classification.  


### Example

```
# import necessary modules
import serket as srk
import mlda
import numpy as np

data = np.loadtxt( "data.txt" )  # load data
data_category = np.loadtxt( "category.txt" )  # load correct labels

# define the modules
obs = srk.Observation( data ) # send the observation to the connected module
mlda1 = mlda.MLDA( 10, catogory=data_category )  # classify into ten classes

# construct the model
mlda1.connect( obs )  # connect obs to mlda1

mlda1.update()  # train mlda
```
