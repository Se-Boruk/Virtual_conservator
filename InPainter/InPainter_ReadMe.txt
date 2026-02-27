
### Inpainter Scripts Overview

Compare_models.py: 
This script performs a quantitative benchmark, pitting different models against each other using PSNR, LPIPS, and loss metrics to determine which one restores images best.

Inpainter_functions.py:
This file houses the logic for loss calculations (like SSIM and Total Variation) and the visualizer class that generates progress plots during training.

InPainter_main.py:
 The command center of the project, responsible for orchestrating the entire training pipeline, data loading, and tracking all experiments via Comet ML.

Upload_to_comet.py:
 A handy utility that sweeps up any local experiment logs and pushes them to Comet ML to ensure remote tracking stays synced.

Architectures.py: 
The brain of the operation, containing the definitions for the various neural network structures—specifically the different Encoder and Decoder versions used throughout the project.

##The Data:

class_map.json: 
Acts as the lookup table for clustering system, translating various artistic styles into the numerical identifiers the model needs for classification.
Classes numbers are used only to train the MaxClass model - the one which uses real information of the classes and it serves as the benchmark to see 
if ground truth class information is better compared to no information or classes info gathered in unsupervised way.


model_evaluation_scores.json:
 A generated record containing the quantitative performance data (loss, PSNR, LPIPS), for the model score evaluation.

