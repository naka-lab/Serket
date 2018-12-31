---
layout: default
---
## Markov Model (MM)

```
mm.MarkovModel( num_samp=100, name="mm", mode="learn" )
```

The hidden Markov model (HMM) can be constructed by connecting `mm.MarkovModel` with the module for classification.
It computes the transition probabilities using the received probabilities and sends the probabilities modified based on the transition probabilities to the connected module.


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| num_samp  | int | Number of sampling iterations |
| name      | string | Module name |
| mode      | string | Choose from learning mode ("learn") or recognition mode ("recog") |


### Methods

- .connect()  
This method connects this module to another module and constructs the model.
- .update()  
This method estimates model parameters and computes probabilities.
The module estimates model parameters in "learn" mode and predicts classes of novel data in "recog" mode.
If training is successful, then the `module{n}_mm` directory is created.
The following files are saved in the directory ({mode} contains the selected mode (learn or recog)):
    - `model.pickle`: The model parameters.
    - `msg_{mode}.txt`: The probabilities that each data element is classified into a class.
    - `trans_prob_learn.txt`: The transition probabilities computed in the learning phase.  


### Example

```
# import necessary modules
import serket as srk
import gmm
import mm
import numpy as np

data = np.loadtxt( "data.txt" )  # load data
data_category = np.loadtxt( "category.txt" )  # load correct labels

# define the modules
obs = srk.Observation( data )  # send the observation to the connected module
gmm1 = gmm.GMM( 10, catogory=data_category )  # classify into ten classes
mm1 = mm.MarkovModel()

# construct the model
gmm1.connect( obs )  # connect obs to gmm1
mm1.connect( gmm1 )  # connect gmm1 to mm1 (construct HMM)

# optimize the model
for i in range(5):
    gmm1.update()  # train gmm1
    mm1.update()  # train mm1
```
