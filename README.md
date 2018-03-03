# Description
This package implements the computation of (improved) Fisher Vectors for video or image datasets. Since the implementation supports the FV computation based on several feature extractor types (currently supporting ResNet, VGG, SIFT and SURF), it allows for comparing one's approach not just to FV but also to common computer vision feature extractors.
In particular, the package covers the following features:
1) extraction, exporting and restoring of several features from videos or image data
2) training a GMM for a Fisher Vector (FVGMM) encoding based on the features, then exporting the FVGMM model parameters
3) computation of (improved) Fisher Vectors from the FVGMM and features

For representation of datasets, we use the convenient and straightforward GulpIO storage format. The above mentioned feature extractors are ready-to-use with this package. 

# Installation
```
$ pip install py-video-features
```

# First steps
##### Setting up the dataset
It is very straightforward to bring your dataset into the right 'gulp' format. The GulpIO documentation gets you started quickly [1]. If you're using one of the prevalent datasets, e.g. ActivityNet, Kinetics or TwentyBN-something-something, it's even simpler to get you started - simply use the available adapter [2] to gulp your local files. 

##### 

we set up both dataset and extractor
```
self.dataset = TwentyBNDataset(batch_size=20, train_dir=self.train_dir, valid_dir=self.valid_dir).getDataLoader(
                                                                                      train=False)
```



# Parameters
Todo: describe config file fields

# Example

A full example can be viewed in the following gist:

https://gist.github.com/ferreirafabio/60323a87ba80c052ab272ff769149577

As shown in the example, we are using `ArmarDataset` class as our data. This is a custom gulpio class that we've written for our own data and which inherits an adapter class from the gulpio package. GulpIO provides the possibility to work directly with familiar datasets such as ActivityNet or TwentyBN-something-something without any implementation efforts but also allows you to 'gulp' your own dataset, which is what we've done in the example case. Please refer to the gulpio package [1] for further information about gulping your own dataset. By having specified in the config.ini the paths to our gulped dataset, we're able to rapidly set-up the pipeline and begin to extract ResNet (or many other) features from the data and train a Fisher Vector GMM to then compute the fisher vectors based on our video features.

[1] https://github.com/TwentyBN/GulpIO
[2] https://github.com/TwentyBN/GulpIO/blob/master/src/main/python/gulpio/adapters.py
