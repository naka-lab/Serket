---
layout: default
---
## HAC (Histogram of Acoustic Co-occurrences)

```
hac.HACFeatureExtractor( filenames, ks, lags=[5,2], name="HACFeatureExtracter", **mfcc_params )
```

`hac.HACFeatureExtractor` is a module for feature extractor. 
It converts wav format data to hac features.
[Click here](https://www.isca-speech.org/archive/interspeech_2008/i08_2554.html) for details about hac.

  
### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| filenames | array | Paths to wav data |
| ks        | array | Number of elements in each code book (triple of int) |
| lags      | array | List of lags to use for calculation (the corresponding histograms are concatenated) |
| name      | string | Name of module |
|mfcc_params| tuple | Parameters for converting to mfcc features (use librosa)  default  "n_mfcc": 13  "n_fft": 2048  "hop_length": 512  "n_mels": 128 |

  
### Example

```
# import necessary modules
import hac
import mlda

# make a list of some paths to wav data
wavs = ["./data00.wav", "./data01.wav", "./data02.wav", 
        "./data03.wav", "./data04.wav", "./data05.wav",
        "./data06.wav", "./data07.wav", "./data08.wav"]

# define the modules
obs = hac.HACFeatureExtractor( wavs, [10,10,10], lags=[5] ) # convert wav data to hac features
mlda1 = mlda.MLDA( 3, [200], category=[0,0,0,1,1,1,2,2,2] )
    
# construct the model
mlda1.connect( obs ) # connect obs to mlda1

mlda1.update() # training mlda1

```