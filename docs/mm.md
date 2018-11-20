---
layout: default
---
## MM (Markov Model)

```
mm.MM( num_samp=100, name="mm", mode="learn" )
```

HMM (Hidden Markov Model) can be constructed by connecting with MM and the module for classification.
It calculates the transition probabilities using the sent probabilities, 
and sends the probabilityies in consideration of the transition probabilities to the connected modules.

  
### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| num_samp  | int | Number of sampling |
| name      | string | Name of module |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |

  
### Methods

- .connect()  
This method connects the module to observations or modules and constructs the model.
- .update()  
This method estimates model parameters and calculates probabilities and others.
The module estimates model parameters in "learn" mode and predict unknown data in "recog" mode.
If learning is successful, the `module {n} _mm` directory is created.
The following files are saved in the directory.({mode} contains the selected mode (learn or recog))
    - `model.pickle`: The model parameters are saved.
    - `msg_{mode}.txt`: The probabilities that each data is in each class are saved.
    - `trans_prob_learn.txt`: The transition probabilities calculated at learning are saved.