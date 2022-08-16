# AN IN-DEPTH EVALUATION OF THE EFFECTS OF IMAGE QUALITY AND DATA BIAS WITHIN SKIN LESION DATASETS
Welcome to my MSc Data Science project for CSC8639.

## Requirements

The requirements can be found in the "requirements.txt" file.
It is ran as part of the jupyter notebook, or you can install manually using `pip install -r requirements.txt`

## Data

### Downloading the Data

The ISIC Challenge Dataset from **2019** should be downloaded from here https://challenge.isic-archive.com/data/#2019.

All of the training data including images, metadata and ground truth should be downloaded. This data will make up our whole dataset for this project.

### Data file structure

The sklearn...
The folders should be unpacked into the following structure where `skin_lesion_data` is the root directory of the source code.

```
skin_lesion_data\ISIC_2019_v2_prepro
├───test
│   ├───AK
│   └───BCC
│   └───BKL
│   └───DF
│   └───MEL
│   └───NV
│   └───SCC
│   └───VASC
├───train
│   ├───AK
│   └───BCC
│   └───BKL
│   └───DF
│   └───MEL
│   └───NV
│   └───SCC
│   └───VASC
├───val
│   ├───AK
│   └───BCC
│   └───BKL
│   └───DF
│   └───MEL
│   └───NV
│   └───SCC
│   └───VASC
```


<!-- - `ISIC-2017_X__Part1_GroundTruth/` contains segmentation maps (used only in EDA) of the skin lesions for each input image
- `ISIC-2017_X_Data/` contains the raw, input images to be classified
- `ISIC-2017_X_Data_metadata.csv` contains patient information for each input image in the corresponding `ISIC-2017_X_Data`
- `ISIC-2017_X_Part3_GroundTruth.csv` contains the classification labels for two classification tests. We only use the `image_id` and `melanoma` columns -->

## Part 1: Image Quality

1. "quality_assessment_models_file_reorganising.ipynb"
The downloaded ISIC 2019 training dataset is split into train, validation and test splits using sklearn "train_test_split" function to an 80:10:10 split.

2. "pre_processing_classes.ipynb"
The images undergo pre-processing. Several technqiues were traillied and the code for each tecqniue is contained in this file.

This file contains several classes, for image preprocessing. Including:
- HairRemoval(): hair removal.
- CropBlackCircle(): cropping strategy to cut large black boundary.
- ShadesOfGrey(): colour adaptation.
- ImageResize(): using the PIL package resize with the bilinear option.

Further, the code used to apply the final chosen pre-processing stages (hair removal and cropping stragey) to all images is included.

3. "data_quality_assessment.ipynb"

First stage was to assess the data quality, in order to determine whether exploring data quality within the skin lesion dataset would be a variable investigation. To do this we used the PyTorch Image Quality (PIQ) package, further information available at: https://github.com/photosynthesis-team/piq and package documentation at: https://piq.readthedocs.io/en/latest/functions.html

We have initially chosen, Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) and Total Variation, as both of these are No Reference Metrics. The exploration of uaing BRISQUE and TV is found in this file.

We chose BRISQUE as our IQA. We calculated the scores for the whole dataset, after image pre-processing had taken place.

4. "quality_models.ipynb"

This file contains the script for multi-class classification. This includes custom classes, such as ImageResize() and the several Custom data samplers tested in this project.

It includes the pre-processing steps, such loading the data and augmentations.

In order todevelop the model we optimised the GPU batch size to 256....MENTION OTHER THINS OPTIMISER AND SUCH

We compared the performance of using to technqiues for class imbaance, including the Imabalnced Dataset Sampler (link to repo) and weighted cross-entropy.

The network is built using DenseNet architecture, and then trained.

The model includes validation and is lastly tested.

Weights and Biases is used to monitor all metrics (F1, balanced accuracy, precision and recall).

5. "graphs_BRISQUE_vs_random_results.ipynb"

The results from wandb were downloaded to produce the final graphical representations for balanced accuracy an F1 for the final dissertation report.

## Part 2: Data Bias

1. "skin_colour.ipynb"

 We calculated the skin colour using the method described by Bevan et al.  https://arxiv.org/pdf/2202.02832.pdf using code supplied in the authors publicly available github repository https://github.com/pbevan1/Detecting-Melanoma-Fairly

 We then, combined the skin tone to the existing ISIC metadata in one .csv file.

2. "8_class_model_data_bias_results.ipynb"

Initially, the same multi-class classification model as used for Part 1 Image Quality was used. However, after analysing the results, we noticed that many of the categories within the metadata did not have any data.

Therefore, we re-evaluted our method and went on to design a binary classification model.

3. "data_balance.ipynb"

We performed Exploratory Data Analysis (EDA) to explore the metadata. This file explores the metadata variables and their categories. This initially was comprised of the following:

- Lesion Class type: 8 classes including melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis (AK), benign keratosis (BKL), dermatofibroma (DF), vascular lesion (VASC), squamous cell carcinoma (SCC).
- Gender: contained two categories; male and female.
- Age: a group for the nearest 5th year from 0 to 85 years.
- Skin tone: six fitzpatrick skin tones, from 1-6.
- Skin lesion anatomical location: contained eight categories, including anterior torso, head/neck, lateral torso, lower extremity, oral/genital, palms/soles, posterior torso and upper extremity.

After data manipluation the metadata variables were the same except for the Skin lesion class and age, which were now structured as:

- Class: melanoma (MEL) and non-melanoma (NONMEL).
- Age: four categories; 0-20, 21-40, 41-60 and >60 years.

EDA has been performed in both cases, for the whole dataset as well as for each split, train, validation and test to ensure each split was representative of the whole dataset.

4. "bias_model_file_reorganising.ipynb"

In order to approach our classifcation as a binary rather than multi-class, we re-organised the skin lesion classes.
To do so we made a copy of the data, and reorganised each of the folders (train, val and test) so that within each folder the 7 non-melanoma class subfolders are merged, so that it results in two folders MEL and NONMEL.

5. "bias_binary_model.ipynb"

This file conains the stages for binary-class classification. It is simialar and uses the same hyperparameters as for the multi-class classification used in Part 1, however instead handles only two classes. The data folder has already been pre-organised to contain two folders for the two classes, NON-MEL and MEL.

This file also contains a custom image sampler that allows for different ratios of female:male images to be sampled, this was created to test the performance on the model by using imbalanced geneder ratios, we tested equal numbers of female:male images of 6000:6000 and then with 2000:10000 images.

6. "binary_model_data_bias_results.ipynb"

This file includes the results from binary classification for gender, skin tone, age and anatomical skin lesion location.

The metrics included, balanced accuracy, F1, precision and recall.

Statistical testing including Spearmans rank correlation and Chi-squared testing was also used alongside graphical representation of results.

 # Code running instructions
 The code is as automated as possible, within the image_classification.ipynb file under '2.0 Configuration dictionary and Custom Classes', the individual models relating to each experiment can be modified depending on which experiment you want to recreate.
 Additionally, if you have an account with Weights and Biases (https://wandb.ai/site), then set 'use_wandb' to TRUE and enter your username in the 'entity' variable. Once the following configurations are made, run the entire notebook.