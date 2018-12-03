---
layout: default
---
## MLDA (Multimodal Latent Dirichlet Allocation)

```
mlda.MLDA( K, weights=None, itr=100, name="mlda", category=None, mode="learn" )
```

`mlda.MLDA` is a module for unsupervised classification.
It calculates the probabilities that each data is classified into each class and each feature of each modality is generated in each data,
and sends them to the connected modules.


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| K         | int | Number of clusters |
| weghits   | array | Weight for each modality |
| itr       | int | Number of iteration |
| name      | string | Name of module |
| category  | array | Correct class labels |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |


### Methods

- .connect()  
This method connects the module to observations or modules and constructs the model.
- .update()  
This method estimates model parameters and calculates probabilities and others.
The module estimates model parameters in "learn" mode and predict unknown data in "recog" mode.
If training is successful, the `module {n} _mlda` directory is created.
The following files are saved in the directory.({mode} contains the selected mode (learn or recog))
    - `model.pickle`: The model parameters are saved.
    - `acc_{mode}.txt`: The accuracy calculated when category is given is saved.
    - `categories_{mode}.txt`: The categories in which each data is classified are saved.
    - `Pdz_{mode}.txt`: The probabilities that each data d is in each class z are saved.
    - `Pmdw[i]_{mode}.txt`: The probabilities that each feature w of each modality i is generated in each data d are saved.  


### Example

```
# import necessary modules
import serket as srk
import mlda
import numpy as np

data = np.loadtxt( "data.txt" )  # load a data
data_category = np.loadtxt( "category.txt" )  # load a correct label

# define the modules
obs = srk.Observation( data ) # send the observation to the connected module
mlda1 = mlda.MLDA( 10, catogory=data_category )  # classify into ten classes

# construct the model
mlda1.connect( obs )  # connect obs to mlda1

mlda1.update()  # train mlda
```
