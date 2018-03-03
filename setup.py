from setuptools import setup

setup(name="videofeatures",
      version='0.1',
      description='Feature extraction from video or image (ResNet, VGG, SIFT, SURF) and training of a Fisher Vector GMM to compute (improved) '
                  'Fisher Vectors',
      url='https://github.com/jonasrothfuss/python-image-video-features',
      author='Jonas Rothfuss, Fabio Ferreira',
      author_email='fabioferreira@mailbox.org',
      license='MIT',
      packages=['videofeatures'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
        'numpy',
        'scikit_learn'
        'py-fisher-vector'
        'keras'
        'pandas'
        'opencv_python'
        'gulpio'
      ],
      zip_safe=False)