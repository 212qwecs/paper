# Prediction of Homologous Protein Thermostability at the Single-cell Level by Incorporating Explicit and Implicit Sequence Features
This repository contains code and models for the paper "Predicting Thermostability of Homologous Protein Pairs Through Explicit and Latent Feature Fusion". The study predicts temperature difference stability between homologous protein pairs using machine learning approaches with feature fusion.
Key Features
​Dual Feature Extraction:
-Explicit features: Extracted using iLearnPlus (not included)
-Latent features: Learned through state-of-the-art protein language models (BioBERT, ProtBERT, ESM-1b, ESM-2, etc.)

​Two-Task Framework:
-📊 Classification: Binary prediction of thermophilic vs. mesophilic proteins
-🔢 Regression: Temperature difference prediction between homologous pairs

​Hierarchical Modeling:
Baseline models using individual feature types (SVM/RF)
Performance-driven feature fusion
Enhanced prediction through fused feature space
