This project presents a deep-learning-based approach to detect visual-binary sources within 2.5 arcseconds using stamp images from multiple astronomical surveys. We build and train a convolutional neural network (CNN) to classify whether a given stamp image contains a visual-binary system.

## Project Overview

We use stamp images of 27,730 sources from the Orion catalog, combining data from both all-sky (e.g., 2MASS, WISE) and partial-sky surveys (e.g., Spitzer, HST, JWST).

## Stacked Image Preparation

Images from each survey are stacked across multiple filters (e.g., J/H/K for 2MASS, g/r/i/y/z for Pan-STARRS).
High-resolution surveys (HST, JWST) resolve binaries clearly; lower-resolution surveys (e.g., 2MASS, ZTF) do not.
## Example images
![00222](https://github.com/Ilknur-Gezer/visual-binary-detection/blob/main/00222.png?raw=true)
Stacked images of 00222 from different surveys are displayed. From upper left to bottom right stacked images of
2MASS, ZTF, neoWISER, unWISE, SDSS9, and PanSTARRS, respectively.


Unresolved binary: 2MASS / ZTF
See /examples/ folder for sample images demonstrating visible vs. unresolved binary sources.
## Data Annotation
Each image is manually labeled as either:
Binary (1) — visual-binary source resolved within 2.5"
Non-Binary (0) — binary not resolved or absent
Training set includes:
~150 binary / 150 non-binary (group 3)
~40 binary / 40 non-binary (group 4)
## Model Architecture & Training
Based on ResNet34, implemented in PyTorch
Custom layers added for better feature extraction
No pre-trained weights; trained from scratch
Training settings:
40 epochs
GPU acceleration
70/15/15 split for train/test/validation
## Evaluation
Performance is evaluated on independent test sets.
Results show successful identification of binaries, especially in high-resolution surveys.
See Table 2 in the paper/report for classification metrics.
## Deployment
Trained model can now classify new stamp images and detect visual-binary sources from Pan-STARRS, SDSS9, and HST.
Easily expandable to other surveys with minimal retraining.

