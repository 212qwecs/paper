# Prediction of Homologous Protein Thermostability at the Single-cell Level by Incorporating Explicit and Implicit Sequence Features
This repository contains code and models for the paper "Predicting Thermostability of Homologous Protein Pairs Through Explicit and Latent Feature Fusion". The study predicts temperature difference stability between homologous protein pairs using machine learning approaches with feature fusion.
## Dataset

### Dataset Source
The protein dataset used in this study is sourced from the research by Leuenberger et al. published in Science (DOI: 10.1126/science.aai7825), which contains protein melting temperature data and corresponding homologous protein pairs. The original dataset consisted of 890 homologous protein pairs, covering a total of 1,127 proteins. During the data screening process, we removed protein sequences marked as obsolete in the database, ultimately retaining 881 valid homologous protein pairs for subsequent analysis. Additionally, the revised manuscript clearly outlines the data acquisition process. To facilitate review and further research, we have compiled the protein IDs, corresponding sequences, and relevant annotations into an Excel file, which is saved in the "data" folder.
## Key Features

### ​Dual Feature Extraction:
- Explicit features: Extracted using iLearnPlus (not included)
- Latent features: Learned through state-of-the-art protein language models (BioBERT, ProtBERT, ESM-1b, ESM-2, etc.),All the related code files are stored in the model/ folder.

## Code and Model

### Installing Dependencies

First, clone this repository:

```bash
git clone https://github.com/212qwecs/paper.git
cd paper
```

Next, install the required dependencies by running:
```
pip install -r requirements.txt
```
This will install all the necessary libraries and frameworks needed for running the code, including those for machine learning, data preprocessing, and visualization.

Running the Code
Training the Model
To train the model using the dataset, run the following command in the root directory of the repository:
```
python train.py --data_path <path_to_data> --output_dir <path_to_save_model>
```
This will:

Load the data from the provided <path_to_data>.

Train the model using the SVM or RF (Random Forest) classifier based on the explicit and latent features.

Save the trained model to the specified <path_to_save_model> directory.

Testing the Model
To evaluate the model on a new dataset or make predictions, run the following command:
```
python test.py --model_path <path_to_trained_model> --data_path <path_to_test_data>
```
This will:

Load the model from <path_to_trained_model>.

Test the model on the dataset at <path_to_test_data> and output the predictions.

Example Usage
Here’s an example of how you might use the training and testing process:
# Training the model
```
python train.py --data_path data/train.csv --output_dir models/
```
# Testing the model
```
python test.py --model_path models/model_best.pth --data_path data/test.csv
```
This will:
Load the model from <path_to_trained_model>.
Test the model on the dataset at <path_to_test_data> and output the predictions.

Code Explanation
train.py: This script is used to train the model. It accepts the training data, processes the features, and uses machine learning algorithms like SVM or RF for model training.

test.py: This script loads a pre-trained model and makes predictions on the test data.

model.py: Defines the model architecture, including the feature fusion process and how explicit and latent features are combined.

utils.py: Contains utility functions for data preprocessing, feature extraction, and evaluation metrics.

## Key Features of the Framework
### ​Two-Task Framework:
- 📊 Classification: Binary prediction of thermophilic vs. mesophilic proteins
- 🔢 Regression: Temperature difference prediction between homologous pairs

### ​Hierarchical Modeling:
Baseline models using individual feature types (SVM/RF)
Performance-driven feature fusion
Enhanced prediction through fused feature space

### Example of Training and Testing Workflow
Data Preprocessing: The data should be preprocessed using the data_preprocessing.py script. This includes extracting explicit features using iLearnPlus and latent features using the protein language models.

Model Training: Train the model using the train.py script, specifying the location of the training data.

Model Evaluation: Once the model is trained, it can be evaluated using the test.py script, which will provide predictions for the test set.
