# Feature Agglomeration in Digit Images

This repository contains a script demonstrating the use of the Feature Agglomeration algorithm applied to a digit image dataset. The script performs dimensionality reduction and visualizes the results before and after applying the algorithm.

## Libraries:

 - matplotlib
 - numpy
 - scikit-learn
   
You can install the required libraries using:
```bash
pip install matplotlib numpy scikit-learn
```

## Script Description

The script performs the following steps:
1. **Loading Data**:
  - Loads the digit dataset from `sklearn.datasets`.
  - Retrieves the digit images for processing.
    
2. **Data Preparation**:
  - Reshapes the images into a one-dimensional vector for analysis.
  - Creates grid-based connectivity from the first image.
    
3. **Feature Agglomeration**:
  - Applies the Feature Agglomeration algorithm (`FeatureAgglomeration`) with 32 clusters.
  - Transforms the original data to a reduced space and then reconstructs the original data from the reduced data.
    
4. **Visualization**:
  - Configures and displays a figure with subplots of the original images, reconstructed images, and cluster labels.
    
## Running the Script

To run the script, simply execute the Python file:
```bash
python feature_agglomeration_digits.py
```
The script will display a figure with subplots showing the original images, the images after feature agglomeration, and the cluster labels.
