# Painting style recognition using traditional image analysis techniques - Project
<img src="https://github.com/Se-Boruk/AoC_project/blob/master/Visuals/Logo.jpg?raw=true" width="780">

This project was part of the Artificial Intelligence & Machine Learning course.

It focuses on creating method for classifying painting styles across images with traditional image analysis techniques. Without using any form of neural networks.

## Project contains scripts:
- **Config.py**:
  > Most important parameters
- **Main.py**:
  > The primary execution script. It manages the training pipeline, including PCA, data scaling, and SVM model fitting
- **Data_analysis.py**:
  > Diagnostic script used to visualize class distributions and histograms to identify dataset imbalances
- **Functions.py**:
  > Library containing the logic for feature extraction (Color, Texture, Shape) and model evaluation helpers
- **RFE.py**:
  > Feature optimization script. Implements Recursive Feature Elimination to rank and prune feature groups for better model efficiency 
- **Model_Validation.py**:
  > Experimental notebook used for analyzing Recursive Feature Elimination (RFE) results and testing stylistic image similarity. Also used for creation of the final report.    

### Due to the file size, project does not contains:
- Training dataset
- Trained models (not as big but can be trained relatively easy trained. (Useless without database anyway as it's mostly isolated project)

## Explanation of concept

Instead of using "black-box" AI, we used shallow learning to decode artistic styles by turning visual features into numerical ones. We manually extracted features like color distributions, brushstroke textures (LBP), and geometric shapes (HOG) to build a unique mathematical "fingerprint" for movements like Impressionism or Cubism. 

These methods allows for good statisctical description of basic concepts, colors, objects which translate into the painting style.

## End Project pipeline (after initial experiments)
- **Data Preprocessing**: Preparing the WikiArt database to optimized format (.arrow) + resizing to 512x512 size. In some cases also filtering data to merge styles under a 2.5% threshold into single class

- **Feature Extraction**: Running a parallel pipeline to extract features from images

- **Optimization**: Using RFE to rank features and PCA to condense the data while keeping 99% of the important info.

- **Model Training**: Teaching an SVM classifier to recognize styles using Nystroem RBF kernel approximation.

- **Testing**: Measuring accuracy and building a tool to find similar-looking paintings


## Results
The model recognized the exact style correctly **37.7%** of the time, but it was remarkably accurate at getting the correct style into its "top five" list—reaching nearly **80%**. 

### Confusion matrix ; Final model
<img src="https://github.com/Se-Boruk/AoC_project/blob/master/Visuals/SVM_full_scores.png?raw=true" width="780">

### Accuracy scores ; Final model

<img src="https://github.com/Se-Boruk/AoC_project/blob/master/Visuals/Top_n_accuracy.png?raw=true" width="780">

We also found that **edge statistics feature extractor** is the strongest giveaway for art styles, while some other extractors especially these based on the image frequency attributes were not as much helpful.

## Output vector similarity
One of the additional tasks inside the project was to find similar paintings. 

We calculate the Euclidean distance between their feature vectors. If two images have similar feature vectors, that means they most likely are visually similar to each other.

The system identified the closest neighbours in this space to show which paintings are statistically related, regardless of what style they actually represent.

### Set of 5 nearest neighbours to the random image

<img src="https://github.com/Se-Boruk/AoC_project/blob/master/Visuals/Similar_imgs_1.jpg?raw=true" width="780">

<img src="https://github.com/Se-Boruk/AoC_project/blob/master/Visuals/Similar_imgs_2.jpg?raw=true" width="780">

## Details

To acces more information you can view the **report** or the **presentation** which are located in the **"Report"** folder

## Contributors
**Sebastian Borukało (Se-Boruk)**: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation 

**Jakub Bukała (Kuba917)**: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation

**Jakub Smakowski (CodeExplorerPL)**: Conceptualization, Methodology, Validation, Formal Analysis, Writing - Original Draft, Writing - Review & Editing, Visualization









