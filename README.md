\# Applied Machine Learning – Audio Classification Project



This repository contains a complete end-to-end \*\*Applied Machine Learning (AML)\*\* project for classifying audio clips using MFCC features and multiple machine learning models.  

All code is structured into reusable modules (`utils.py`, `models.py`, `main.py`) and runs \*\*fully dynamically\*\* without hardcoded paths.



---



\##  Project Structure



project-root/

│

├─ data/ # Place all audio files here (e.g., WAV/MP3)

├─ output/ # Auto-generated: processed CSVs, model results

│

├─ main.py # Full pipeline execution

├─ utils.py # Audio loading, splitting, MFCC extraction, encoding

├─ models.py # ML models, NN, kNN sweep, RF, ensemble

│

├─ README.md

└─ requirements.txt





---



\##  Features



\###  Audio Processing

\- Automatic reading of all audio files in `./data`

\- Splitting audio into fixed-size 2-second clips

\- MFCC feature extraction (13 coefficients)

\- Mean aggregation per MFCC over time



\###  Feature Engineering

\- MFCC → DataFrame expansion (`mfcc1`…`mfcc13`)

\- Label encoding + one-hot encoding

\- Correlation matrix visualization

\- Dropping low-correlation features

\- Histogram plotting for distributions



\###  Machine Learning Models

\- \*\*Random Forest Classifier\*\*

\- \*\*k-Nearest Neighbors\*\*

&nbsp; - Auto sweep across odd k from 3 to 17

\- \*\*Neural Network (Keras)\*\*

&nbsp; - 13 → 64 → 32 → 7 architecture

&nbsp; - Early stopping at target accuracy (79%)

\- \*\*Weighted Hard Voting Ensemble\*\*

&nbsp; - Combines RF + NN + kNN

&nbsp; - Default weights: RF=3, NN=3, kNN=2



---



\##  How to Run



\### 1. Install dependencies

```bash

pip install -r requirements.txt





