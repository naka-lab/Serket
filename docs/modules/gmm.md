---
layout: default
---
## Gaussian Mixture Model (GMM)

```
gmm.GMM( K, itr=100, name="gmm", category=None, mode="learn" )
```

`gmm.GMM` is a module for unsupervised classification based on a Gaussian mixture model.
It computes the probabilities that each data element is classified into each class and the means of the distributions and sends them to the connected modules.


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| K         | int | Number of clusters |
| itr       | int | Number of iterations |
| name      | string | Module name |
| category  | array | Correct class labels |
| mode      | string | Choose from learning mode ("learn") or recognition mode ("recog") |


### Methods

- .connect()  
This method connects this module to an observation or a module and constructs the model.
- .update()  
This method estimates model parameters and computes probabilities.
The module estimates model parameters in "learn" mode and predicts classes of novel data in "recog" mode.
If training is successful, then the `module{n}_gmm` directory is created.
The following files are saved in the directory ({mode} contains the selected mode (learn or recog)):
    - `model.pickle`: The model parameters.
    - `acc_{mode}.txt`: The accuracy computed if the optional argument `category` is set.
    - `class_{mode}.txt`: The classes into which each data element is classified.
    - `mu_{mode}.txt`: The means of the distributions of each class.
    - `Pdz_{mode}.txt`: The probabilities that each data element is classified into a class.  


### Example

```
# import necessary modules
import serket as srk
import gmm
import numpy as np

data = np.loadtxt( "data.txt" )  # load data
data_category = np.loadtxt( "category.txt" )  # load correct labels

# define the modules
obs = srk.Observation( data )  # send the observation to the connected module
gmm1 = gmm.GMM( 10, catogory=data_category )  # classify into ten classes

# construct the model
gmm1.connect( obs )  # connect obs to gmm1

gmm1.update()  # train gmm1
```
