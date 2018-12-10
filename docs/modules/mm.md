---
layout: default
---
## MM (Markov Model)

```
mm.MarkovModel( num_samp=100, name="mm", mode="learn" )
```

HMM (Hidden Markov Model) can be constructed by connecting with `mm.MarkovModel` and the module for classification.
It calculates the transition probabilities using the received probabilities,
and sends the probabilities modified based on the transition probabilities to the connected modules.


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| num_samp  | int | Number of iteration of sampling |
| name      | string | Name of module |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |


### Methods

- .connect()  
This method connects the module to observations or modules and constructs the model.
- .update()  
This method estimates model parameters and calculates probabilities.
The module estimates model parameters in "learn" mode and predicts classes of novel data in "recog" mode.
If training is successful, the `module {n} _mm` directory is created.
The following files are saved in the directory.({mode} contains the selected mode (learn or recog))
    - `model.pickle`: The model parameters.
    - `msg_{mode}.txt`: The probabilities that each data is classified into each class.
    - `trans_prob_learn.txt`: The transition probabilities calculated in the learning phase.  


### Example

```
# import necessary modules
import serket as srk
import gmm
import mm
import numpy as np

data = np.loadtxt( "data.txt" )  # load a data
data_category = np.loadtxt( "category.txt" )  # load a correct label

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
