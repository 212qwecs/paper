# Prediction of Homologous Protein Thermostability at the Single-cell Level by Incorporating Explicit and Implicit Sequence Features
This repository contains code and models for the paper "Predicting Thermostability of Homologous Protein Pairs Through Explicit and Latent Feature Fusion". The study predicts temperature difference stability between homologous protein pairs using machine learning approaches with feature fusion.
## Dataset

### Dataset Source
The protein dataset used in this study is sourced from the research by Leuenberger et al. published in Science (DOI: 10.1126/science.aai7825), which contains protein melting temperature data and corresponding homologous protein pairs. The original dataset consisted of 890 homologous protein pairs, covering a total of 1,127 proteins. During the data screening process, we removed protein sequences marked as obsolete in the database, ultimately retaining 881 valid homologous protein pairs for subsequent analysis. Additionally, the revised manuscript clearly outlines the data acquisition process. To facilitate review and further research, we have compiled the protein IDs, corresponding sequences, and relevant annotations into an Excel file, which is saved in the "data" folder.
## Key Features

### ​Dual Feature Extraction:
- Explicit features: Extracted using iLearnPlus (not included)
- Latent features: Learned through state-of-the-art protein language models (BioBERT, ProtBERT, ESM-1b, ESM-2, etc.)

## ​Two-Task Framework:
- 📊 Classification: Binary prediction of thermophilic vs. mesophilic proteins
- 🔢 Regression: Temperature difference prediction between homologous pairs

​Hierarchical Modeling:
Baseline models using individual feature types (SVM/RF)
Performance-driven feature fusion
Enhanced prediction through fused feature space
