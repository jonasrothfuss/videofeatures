# Description
This package implements the computation of (improved) Fisher Vectors for video or image datasets. The implementation supports the computation based on several several feature extractor types (ResNet, VGG, SIFT and SURF) and for example allows to compare one's method to a common computer vision baseline.
In particular, the package covers the following features:
1) extraction, exporting and restoring of features from videos or image data
2) training a GMM for a Fisher Vector (FVGMM) encoding based on the features, then exporting the FVGMM model parameters
3) computation of (improved) Fisher Vectors from the FVGMM and features


# Installation



# Parameters
Todo: describe config file fields

# Example

A full example can be viewed in the following gist:

https://gist.github.com/ferreirafabio/60323a87ba80c052ab272ff769149577

As shown in the example, we are using `ArmarDataset` class as our data. This is a custom gulpio class that we've written for our own data and which inherits an adapter class from the gulpio package. GulpIO provides the possibility to work directly with familiar datasets such as ActivityNet or TwentyBN-something-something without any implementation efforts but also allows you to 'gulp' your own dataset, which is what we've done in the example case. Please refer to the gulpio package [1] for further information about gulping your own dataset. By having specified in the config.ini the paths to our gulped dataset, we're able to rapidly set-up the pipeline and begin to extract ResNet (or many other) features from the data and train a Fisher Vector GMM to then compute the fisher vectors based on our video features.

[1] https://github.com/TwentyBN/GulpIO
