# AN IN-DEPTH EVALUATION OF THE EFFECTS OF IMAGE QUALITY AND DATA BIAS WITHIN SKIN LESION DATASETS
Welcome to my MSc Data Science project for CSC8639.

## Requirements

The requirements can be found in the "requirements.txt" file.
It is ran as part of the jupyter notebook, or you can install manually using `pip install -r requirements.txt`

## Data

### Downloading the Data

The ISIC Challenge Dataset from **2019** should be downloaded from here https://challenge.isic-archive.com/data/#2019.

The training data for tasks 1 and 2 along with the training ground truth should be downloaded. This data will make up our whole dataset for this project.

## Part 1: Image Quality

### 1. "quality_assessment_models_file_reorganising.ipynb"
The downloaded ISIC 2019 training dataset is split into train, validation and test splits using sklearn "train_test_split" function to an 80:10:10 split. Within each split, 8 folders are created, one for each skin lesion class. The images are then allocated to the correct folder using the class name within the image name. The final folder structure is organised in the following way:

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


### 2. "pre_processing_classes.ipynb"
The images undergo pre-processing. Several techniques were trialed and the code for each technique is contained in this file.

This file contains several classes, for image preprocessing. Including:
- HairRemoval(): hair removal.
- CropBlackCircle(): cropping strategy to cut large black boundary.
- ShadesOfGrey(): colour adaptation.
- ImageResize(): using the PIL package resize with the bilinear option.

Further, the code used to apply the final chosen pre-processing stages (hair removal and cropping stragey) to all images is included, ShadesOfGrey() was not used.

### 3. "data_quality_assessment.ipynb"

We first trialed several methods to determine image quality. To do this we used the PyTorch Image Quality (PIQ) package, further information available at: https://github.com/photosynthesis-team/piq and package documentation at: https://piq.readthedocs.io/en/latest/functions.html

We have initially chosen, Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) and Total Variation (TV), as both of these are No Reference Metrics. The exploration of using BRISQUE and TV is found in this file.

We chose BRISQUE as our IQA. We calculated the scores for the whole dataset, after image pre-processing had taken place.

### 4. "quality_models.ipynb"

This file contains the script for training the multi-class classification model. This includes custom classes, such as ImageResize() and the several Custom data samplers tested in this project.
Several hyperparameters were tested such as batch size, epochs and different schedulers.
We compared the performance of using two techniques for class imbalance, including the Imbalanced Dataset Sampler (https://github.com/ufoym/imbalanced-dataset-sampler) and weighted cross-entropy.
We also trialed various augmentation techniques including automated augmentation.

The network was built using DenseNet architecture.

Weights and Biases is used to monitor all metrics (F1, balanced accuracy, precision and recall).
If you have an account with Weights and Biases (https://wandb.ai/site), then set 'use_wandb' to TRUE and enter your username in the 'entity' variable. Once the following configurations are made, run the entire notebook.

### 5. "graphs_BRISQUE_vs_random_results.ipynb"

The results from wandb were downloaded to produce the final graphical representations for balanced accuracy an F1 for the final dissertation report.

### 6. "ISIC_2019_Training_Metadata_with_full_paths_with_brisque_and_class.csv"
This .csv contains the original .csv data from the metadata file downloaded from ISIC 2019, this includes the metadata variables.
Further information from our work was also added this included; skin tone, image BRISQUE score, skin lesion class, full image path and data split (train, validation or test).


## Part 2: Data Bias

### 1. "skin_colour.ipynb"

 We calculated the skin colour using the method described by Bevan et al.  https://arxiv.org/pdf/2202.02832.pdf using code supplied in the authors publicly available github repository https://github.com/pbevan1/Detecting-Melanoma-Fairly

 We then, combined the skin tone to the existing ISIC metadata in one .csv file.

### 2. "8_class_model_data_bias_results.ipynb"

Initially, the same multi-class classification model as used for Part 1 Image Quality was used. However, after analysing the results, we noticed that many of the categories within the metadata did not have any data.

Therefore, we re-evaluated our method and went on to design a binary classification model.

### 3. "data_balance.ipynb"

We performed Exploratory Data Analysis (EDA) to explore the metadata. This file explores the metadata variables and their categories. This initially was comprised of the following:

- Lesion Class type: 8 classes including melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis (AK), benign keratosis (BKL), dermatofibroma (DF), vascular lesion (VASC), squamous cell carcinoma (SCC).
- Gender: contained two categories; male and female.
- Age: a group for the nearest 5th year from 0 to 85 years.
- Skin tone: six Fitzpatrick skin tones, from 1-6.
- Skin lesion anatomical location: contained eight categories, including anterior torso, head/neck, lateral torso, lower extremity, oral/genital, palms/soles, posterior torso and upper extremity.

After data manipulation the metadata variables were the same except for the Skin lesion class and age, which were now structured as:

- Class: melanoma (MEL) and non-melanoma (NONMEL).
- Age: four categories; 0-20, 21-40, 41-60 and >60 years.

EDA has been performed in both cases, for the whole dataset as well as for each split, train, validation and test to ensure each split was representative of the whole dataset.

### 4. "bias_model_file_reorganising.ipynb"

In order to approach our classification as a binary rather than multi-class, we re-organised the skin lesion classes.
To do so we made a copy of the data, and reorganised each of the folders (train, val and test) so that within each folder the 7 non-melanoma class subfolders are merged, so that it results in two folders MEL and NON_MEL.

```
skin_lesion_data\ISIC_2019_v2_prepro_binary
├───test
│   ├───MEL
│   └───NON_MEL
├───train
│   ├───MEL
│   └───NON_MEL
├───val
│   ├───MEL
│   └───NON_MEL
```

### 5. "bias_binary_model.ipynb"

This file contains the stages for binary-class classification. It is similar and uses the same hyperparameters as for the multi-class classification used in Part 1, however instead handles only two classes. The data folder has already been pre-organised to contain two folders for the two classes, NON-MEL and MEL.

This file also contains a custom image sampler that allows for different ratios of female:male images to be sampled, this was created to test the performance on the model by using imbalanced gender ratios, we tested equal numbers of female:male images of 6000:6000 and then with 2000:10000 images.

Weights and Biases is used to monitor all metrics (F1, balanced accuracy, precision and recall).
If you have an account with Weights and Biases (https://wandb.ai/site), then set 'use_wandb' to TRUE and enter your username in the 'entity' variable. Once the following configurations are made, run the entire notebook.

### 6. "binary_model_data_bias_results.ipynb"

This file includes the results from binary classification for gender, skin tone, age and anatomical skin lesion location.

The metrics included, balanced accuracy, F1, precision and recall.

Statistical testing including Spearman's rank correlation and Chi-squared testing was also used alongside graphical representation of results.

### 8. "mel_vs_nonmel.csv"
This .csv contains the information as described in "ISIC_2019_Training_Metadata_with_full_paths_with_brisque_and_class.csv" but with adaptations, these include the binary lesion class, either MEL or NON_MEL. In addition, the age group that is now either one of four categories; 0-20, 21-40, 41-60 and >60 years.
