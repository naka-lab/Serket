---
layout: default
---
## GMM (Gaussian Mixture Model)

```
gmm.GMM( K, itr=100, name="gmm", category=None, mode="learn" )
```

GMM is a module for unsupervised classification.
It calculates the probabilities that each data is classified into each class and the mean of the distribution of the classes, 
and sends to the connected modules.

  
### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| K         | int | Number of clusters |
| itr       | int | Number of iteration |
| name      | string | Name of module |
| category  | array | Correct class labels |
| mode      | string | Choose the mode from learning mode("learn") or recognition mode("recog") |

  
### Method

- .connect()  
This method connects the module to observations or modules and constructs the model.
- .update()  
This method estimates model parameters and calculates probabilities and others.
The module estimates model parameters in "learn" mode and predict unknown data in "recog" mode.
If learning is successful, the `module {n} _gmm` directory is created.
The following files are saved in the directory.({mode} contains the selected mode (learn or recog))
    - `model.pickle`: The model parameters are saved.
    - `acc_{mode}.txt`: The accuracy calculated when category is given is saved.
    - `class_{mode}.txt`: The classes in which each data is classified are saved.
    - `mu_{mode}.txt`: The mean of the distribution of each class z in which each data d is classified are saved.
    - `Pdz_{mode}.txt`: The probabilities that each data d is in each class z are saved.