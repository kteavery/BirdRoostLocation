# BirdRoostLocation
Katherine Avery

This project is an extension of kteavery/BirdRoostDetection and the paper [Automated detection of bird roosts using NEXRAD radar data and Convolutional Neural Networks](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.92) by Carmen Chilson and Katherine Avery.

This code was developed for the Oklahoma Biological Survey for a research project. We use machine learning to automate the detection of pre-migratory purple martin roosts, their location, and the radius of the roost in NEXRAD radar data.

## Requires Software
- tensorflow https://www.tensorflow.org/install/
- Keras https://keras.io/
- PyArt http://arm-doe.github.io/pyart/
- Numpy http://www.numpy.org/
- Matplotlib https://matplotlib.org/

## Setting up Amazon Web Services
- In order to access radar files stored on Amazon Web Services you will first need to setup the .boto file with your user credentials.
- See instructions here: https://aws.amazon.com/developers/getting-started/python/

## Installing and software
- git clone git@github.com:kteavery/BirdRoostLocation.git
- cd BirdRoostLocation
- python setup.py install

## Using the software
Many of the scripts in this class will include instructions at the beginning of the document with instructions of how to use them.
