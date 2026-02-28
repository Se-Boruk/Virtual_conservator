# Virtual Conservator - Project
<img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/Project_logo.png?raw=true" width="580">

This project was part of the Artificial Intelligence & Machine Learning course.

It focused on creating creating pipeline for complete painting restoration which included:<br>
**1) Inpainter**: To repair damaged images by filling missing regions<br>
**2) Clusterizer**: To classify images (and use this information to potentially help in reconstruction)<br>
**3) Upscaler:** To upscale restored image up to 4x of its original resolution

## Project overview

#### Datasets used:
- Inpainter / Clusterizer: "Artificio/WikiArt" [ https://huggingface.co/datasets/Artificio/WikiArt ]
- Upscaler: "huggan/wikiart" [ https://huggingface.co/datasets/huggan/wikiart ]

#### Data processing & model training pipeline:
<img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/Pipeline.png?raw=true" width="580">

#### Potential damage masks:
<img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/Damage_masks.png?raw=true" width="420">


For a comprehensive overview of the entire project, we highly encourage reviewing the **Final_presentation** file (currently only in PL)

## Results
Proposed method allows **high-level reconstruction** of damaged painting and general clusterization (however metrics indicates that InPainter did not benefit from class information).


Test set example reconstructions
<img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/test_real_0001.png?raw=true" width="800">


Additionally we can further increase quality of the painting by using **Upscaling model**.
It both increases the raw resolution, as well as it adds details not present previously in the image.


| 256x256 Input | SRCNN x4 Output (1024x204) |
| :---: | :---: |
| <img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/042222-256x256.png?raw=true" width="400" /> | <img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/result_1024x1024.png?raw=true" width="400" /> |

## GUI
For live usage it is possible to run the GUI (GUI.py) and test functionality in live demo. Few images are given in the GUI folder, but you can choose any image format files from your PC. It works well with other type of images than paintings too.

#### Main window
<img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/GUI_preview_1.png?raw=true" width="360">

#### Results of image restoration
<img src="https://github.com/Se-Boruk/Virtual_conservator/blob/master/Assets/GUI_preview_2.png?raw=true" width="560">

## Project is separated into 4 main modules which corresponding files, models etc. can be found in their respective directories:
- **DataBase**:
  > This folder contains database, scripts neccessary to download, process and manage it. All other modules use the methods from this module 

- **Inpainter**:
  > This folder contains scripts responsible for Inpainter training and testing. It contains as well training logs, architectures of models used through the project and trained models themselves

- **Clustering**:
  > This folder contains scripts related to the Clusterizer training as well as trained PCA and KMeans Clusterizers ready to be used

- **Upscaler**:
  > This folder stores jupyter notebook used in Upscaler training and trained Upscaler model


## Contributors
**Sebastian Borukało (Se-Boruk)**: Inpainting, Project management, Data pipeline and processing

**Jakub Bukała (Kuba917)**: Clusterization, GUI / working LIVE Demo

**Jakub Smakowski (CodeExplorerPL)**: Upscaling, Data pipeline and processing









