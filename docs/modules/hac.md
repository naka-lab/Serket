---
layout: default
---
## Histogram of Acoustic Co-Occurrences (HAC)

```
hac.HACFeatureExtractor( filenames, ks, lags=[5,2], name="HACFeatureExtracter", **mfcc_params )
```

`hac.HACFeatureExtractor` is a module for acoustic feature extraction based on the histogram of acoustic co-occurrences (HAC).
It extracts HAC features from wave-format files.
See [here](https://www.isca-speech.org/archive/interspeech_2008/i08_2554.html) for details about HAC.


### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| filenames | array | Paths to wave files |
| ks        | array | Number of elements in each code book (triplet of int) |
| lags      | array | List of lags to compute co-occurrence of acoustic events (the corresponding histograms are concatenated) |
| name      | string | Module name |
|mfcc_params| tuple | Parameters for computing mfcc features (used librosa)<br>default<br>"n_mfcc": 13<br>"n_fft": 2048<br>"hop_length": 512<br>"n_mels": 128 |


### Example

```
# import necessary modules
import hac
import mlda

# make a list of paths to wav data
wavs = ["./data00.wav", "./data01.wav", "./data02.wav",
            "./data03.wav", "./data04.wav", "./data05.wav",
            "./data06.wav", "./data07.wav", "./data08.wav"]

# define the modules
obs = hac.HACFeatureExtractor( wavs, [10,10,10], lags=[5] )  # convert wav data into hac features
mlda1 = mlda.MLDA( 3, [100], category=[0,0,0,1,1,1,2,2,2] )  # classify into three classes

# construct the model
mlda1.connect( obs )  # connect obs to mlda1

mlda1.update()  # train mlda1
```
