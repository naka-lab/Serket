---
layout: default
---
## Observation

```
serket.Observation( data )
```

`serket.Observation` is a module that sends the observation to the connected module.

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| data      | array| Data        |


### Example

```
# import necessary modules
import serket as srk
import mlda

# generate a data element
data = [[10, 2, 1],
            [8, 1, 1],
            [2, 7, 0],
            [1, 11, 2],
            [1, 1, 8],
            [2, 1, 1]]

# define the modules
obs = srk.Observation( data )  # send the observation to the connected module
mlda1 = mlda.MLDA( 3, category=[0, 0, 1, 1, 2, 2] )  # classify into three classes

# construct the model
mlda1.connect( obs )  # connect obs to mlda1

mlda1.update()  # train mlda1
```
