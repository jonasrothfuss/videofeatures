# Description
This package implements the computation of (improved) Fisher Vectors for video or image datasets. Since the implementation supports the FV computation based on several feature extractor types (currently supporting **ResNet**, **VGG**, **SIFT** and **SURF**), it allows for comparing one's approach not just to FV but also to common computer vision feature extractors.
In particular, the package covers the following features:
1) **extraction**, exporting and restoring **of several features from videos** or image data
2) **training a GMM for a Fisher Vector (FVGMM)** encoding based on the features, then exporting the FVGMM model parameters
3) **computation of (improved) Fisher Vectors** from the FVGMM and features

For representation of datasets, we use the convenient and straightforward GulpIO storage format. The above mentioned feature extractors are ready-to-use with this package. 

# Installation
```
$ pip install py-video-features
```

# First steps
### 1. Setting up the dataset
It is very straightforward to bring your dataset into the right 'gulp' format. The GulpIO documentation gets you started quickly [1]. If you're using one of the prevalent datasets, e.g. ActivityNet, Kinetics or TwentyBN-something-something, it's even simpler to get you started - simply use the available adapter [2] to gulp your local files. 

### 2. Initialization
First, we instantiate both dataset (here ActivityNet) and extractor (ResNet)
```
output_dir = "./output" # where all the features, models and logs are exported to
activitynet = ActivityNetDataset(batch_size=20, train_dir=path_train, valid_dir=path_train).getDataLoader(train=True)
resnet = ResNetFeatures()
pipeline = Pipeline(dataset=activitynet, extractor=resnet, base_dir=output_dir)
```

### 3. Feature extraction, GMM training and FV computation
Having initialized the pipeline, we can do
```
features, labels = pipeline.extractFeatures()
fisher_vector_gmm = pipeline.trainFisherVectorGMM(features)
fisher_vectors, labels = pipeline.computeFisherVectors(features=features, labels=labels, fv_gmm=fisher_vector_gmm)
```

### Full example
A full example can be viewed in the following gist:
https://gist.github.com/ferreirafabio/60323a87ba80c052ab272ff769149577

[1] https://github.com/TwentyBN/GulpIO
[2] https://github.com/TwentyBN/GulpIO/blob/master/src/main/python/gulpio/adapters.py
