---
layout: default
---
## Convolutional Neural Network (CNN)

```
CNN.CNNFeatureExtractor( filenames, name="CNNFeatureExtracter" )
```

`CNN.CNNFeatureExtractor` is a module for feature extraction.
It extracts image features using a CNN with caffemodel.


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| filenames | array | Paths to image files |
| name      | string | Name of module |


### Example

```
# import necessary modules
import CNN
import mlda

# make a list of some paths to image data
data = ["./data00.png", "./data01.png", "./data02.png",
            "./data03.png", "./data04.png", "./data05.png",
            "./data06.png", "./data07.png", "./data08.png"]

# define the modules
obs = CNN.CNNFeatureExtractor( data )  # extract image features
mlda1 = mlda.MLDA( 3, [100], category=[0,0,0,1,1,1,2,2,2] )  # classify into three classes

# construct the model
mlda1.connect( obs )  # connect obs to mlda1

mlda1.update()  # train mlda1
```
