## multi-modal
This is supplimental codes for "Multi-modal modeling with Low-Dose CT and clinical information for diagnostic artificial intelligence on Mediastinal Tumors"

## Usage
1. data_prepare.py: pre-process .nrrd and its segmentation data, and save as .pkl. 
2. python main.py --cv --adam (--woagesexsmoking)
3. python main_radiomics_randomforest.py --cv (--randomforest)

## options
- --woagesexsmoking: run without clinical information such as age and sex
- --randomforest: run by randomforest, if not run by PCA+logistic regression

