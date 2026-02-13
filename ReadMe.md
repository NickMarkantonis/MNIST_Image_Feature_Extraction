# MNIST Image Feature Extraction

## Introduction
The primary objective of this project was for me to get my foot in the water and gain some exprerience in feature extraction from images. I tried applying some of the things that where taught to me in the course HY371 - Image Processing such as orientation clustering.

The Project is on the dataset MNIST, I choose this due to it's extensive documentation and ready-to-use form. The task was to extract features that could then be used as vectors to train models on, this was achieved but with a low accuracy.

To be more specific the accuracy came back to ~61%, something much lower then even just straight up training a model on the pixel values alone. This is mainly due to the insufficient or plain wrong features (such as lines, circles etc) that got extracted and leaves much to be promised. Still thought it signifies that the choice was not random (10%) and even at a lower level some meaningfull features got extracted.

## The features
Following are the features that got extracted as well as how they got extracted from the images.

### Basic Features
These basic features consisted of the images mean instensity and standart instensity as well as the intensity of the edges.

### Orientation Clustering
Info based on the edges and their general orientation, we split them in 4 parts:
1. vertical edges
2. diagonal edges (45°)
3. hroizontal edges
4. diagonal edges (135°)

### Contours and Shape Features
Using Hough Tranforms we detect a bit more data on contours such as lines, circles etc.

## Reflection
This project jump started my experience and I will continue building on it and I hope I can achieve at least a accuracy of over 90% over the next few



