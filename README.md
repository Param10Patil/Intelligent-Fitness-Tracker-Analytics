# Intelligent-Fitness-Tracker-Analytics

## **Project Overview**
This project develops a complete machine learning workflow that converts raw IMU sensor signals (Accelerometer & Gyroscope) into an intelligent fitness tracking system. The solution is capable of recognizing different weightlifting exercises and accurately estimating repetition counts with near clinical-level precision.

## **Description**    

The goal of this project is to construct a full machine learning pipeline. It starts with loading, cleaning, and transforming raw CSV data, followed by time-series visualization to explore underlying behavioral patterns. Advanced anomaly detection methods such as Chauvenet’s Criterion and Local Outlier Factor (LOF) are applied to enhance data reliability. This is followed by an extensive feature engineering phase using techniques like low-pass filtering, Principal Component Analysis (PCA), and clustering methods. For prediction tasks, multiple models including Naive Bayes, Support Vector Machines (SVM), Random Forests, and Neural Networks are trained and evaluated to achieve optimal performance. The project concludes with a custom-designed algorithm that detects and counts repetitions automatically with high precision.

##  **Tools & Technologies**

This project utilizes a comprehensive set of tools from the Python data science ecosystem:

* **Programming Language:** Python 3.x
* **Development Environment:** VS Code (Visual Studio Code), Jupyter Notebooks
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Signal Processing (Time-Series):** `scipy` (Used for Butterworth Low-pass filtering and peak detection methods)
* **Machine Learning & Predictive Modeling:** `scikit-learn` (PCA, Random Forest, SVM, K-Means Clustering, Neural Networks)
* **Data Visualization:** `matplotlib`, `seaborn`
---

## **Part 1: Dataset MetaMotion Physical Activity Analysis**
1. **Overview**
The dataset contains **187 raw CSV files** collected using **MetaMotion wearable sensors**. It records detailed motion signals across multiple resistance training exercises, designed specifically for activity recognition and repetition tracking.
2. **Sensor Specifications**
The dataset is generated using two types of inertial sensors:
    * 3-Axis Accelerometer: Measures linear acceleration along $x, y, z$ axes in $g$ units at 12.5Hz.
    * 3-Axis Gyroscope: Measures angular velocity along $x, y, z$ axes in $deg/s$ at 25Hz.
3. **Metadata & Naming Convention**
Each file follows a structured naming format that enables automatic label extraction:
[Participant]-[Label]-[Category][SetNumber]-[SensorType].csv
   - **Participants**: 5 individuals (A, B, C, D, E).
     
   - **Activity Labels**:
       * bench: Bench Press
       * squat: Squat
       * ohp: Overhead Press
       * dead: Deadlift
       * row: Barbell Row
       * rest: Inactive state.
   
   - **Categories (Intensity)**:
       * Heavy Set: High load sessions with 5 repetitions.
       * Medium Set: Moderate load sessions with 10 repetitions.
     
   - **Set Number**: Indicates sequence order of sets for each activity.
4. **Data Features**
Each record includes:
   - epoch (ms) | Timestamp in milliseconds
   - time | ISO formatted datetime | YYYY-MM-DDTHH:MM:SS |
   - x-axis | Sensor measurement on X-axis | $g$ or $deg/s$ 
   - y-axis | Sensor measurement on Y-axis | $g$ or $deg/s$ 
   - z-axis | Sensor measurement on Z-axis | $g$ or $deg/s$
5. **Data Challenges & Objectives**
   - Multi-Frequency Alignment: Combining 12.5Hz and 25Hz signals.
   - Noise Handling: Removing anomalies using statistical approaches like Chauvenet’s Criterion.
   - Feature Extraction: Deriving meaningful representations using PCA and frequency-based methods.

---

## **Part 2: Data Processing & Integration**

1. **Data Aggregation & Metadata Extraction**
A custom pipeline was built to iterate over 187 CSV files and extract important metadata from filenames such as:
   - Participant ID
   - Exercise label
   - Intensity category

2. **Merging & Resampling Strategy**
To align sensor streams, accelerometer and gyroscope data were merged. A resampling process was applied:
   - Frequency: Standardized to 5Hz (200ms interval)
   - Aggregation: Mean for sensor values, last observation for categorical fields  
This approach ensured temporal consistency and reduced computational load.

3. **Dataset Statistics**
   Final dataset contains 9,009 rows and 10 columns.

   A. Class Distribution
      Balanced across 6 activities:
      - OHP | 1,676
      - Bench Press | 1,665
      - Squat | 1,610
      - Deadlift | 1,531
      - Row | 1,417
      - Rest | 1,110

   B. Participant Distribution
      Spread across individuals:
      - A: 2,988
      - E: 2,645
      - C: 1,481
      - D: 1,052
      - B: 843

4. **Data Quality Assurance**
   - Missing values: None
   - Features: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, participant, label, category, set

---

## **Part 3: Data Visualization**

1. **Repetition Patterns**   
Accelerometer plots reveal repeating waveforms corresponding to exercise repetitions. Each peak indicates a completed movement cycle, with patterns varying across exercises.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/participant%20A%20on%20squat.png)

2. **Intensity Comparison**   
Medium-weight sets show faster repetition cycles, while heavy sets display slower and noisier signals due to increased effort.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/heavy_vs_medium_sets_y_acc.png)

3. **Participant Variability**   
Differences across individuals highlight biomechanical variations, reinforcing the need for diverse training data.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/participants%20in%20bench%20set%20(acc%20y).png)

4. **Sensor Contribution**   
Accelerometers capture motion magnitude, while gyroscopes capture rotational dynamics. Combining both improves prediction accuracy.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/Bench%20(D).png)

5. **Rest State Analysis**   
During rest periods, signals remain nearly flat with minimal noise.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/Rest%20(E).png)

---

## **Part 4: Outlier Detection & Handling Strategy**

1. **Detection Methods**
Tested IQR, LOF, and Chauvenet’s Criterion for identifying anomalies.

2. **Selected Approach**
Chauvenet’s Criterion provided the best balance by removing noise without affecting valid motion peaks.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/Chauvenet_ACC_x.png)
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/Chauvenet_gyr_x.png)

3. **Handling Strategy**
Outliers were replaced with NaN to maintain time continuity instead of removing rows.

---

## **Part 5: Feature Engineering & Data Transformation**

1. **Interpolation**
Missing values were filled using interpolation to preserve smooth temporal transitions.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/sample%20with%20nan.png)
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/sample%20after%20filling%20nans.png)

2. **Noise Reduction**
Applied Butterworth low-pass filtering to remove high-frequency noise.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/before%20and%20after%20lowpass%20filter.png)

3. **Dimensionality Reduction**
PCA reduced six sensor dimensions into three principal components.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/elbow%20tech%20for%20PCA.png)
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/PCAs%20plot.png)

4. **Magnitude Features**
Computed orientation-independent magnitude values.

5. **Clustering**
K-Means identified natural groupings in movement patterns.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/elbow%20tech%20for%20KMEANS.png)
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/3D%20visualization%20for%20acc%20data.png)

---

## **Part 6: Predictive Modeling & Algorithm Selection**

1. **Data Splitting**
Used stratified 75/25 split across feature sets.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/train%20test%20split.png)

2. **Feature Selection**
Forward selection identified top-performing features with minimal complexity.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/Forward%20Feature%20Selection.png)

3. **Model Comparison**
Random Forest and Neural Networks achieved highest performance.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/models%20accuracies.png)
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/models%20scores.png)

**Evaluation**

- Random split accuracy: 99.4%  
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/CM%20participant%20A.png)

- Leave-one-subject-out: 98.6%  
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/CM%20for%20complex%20model.png)

---

## **Part 7: Repetition Counting & Final Evaluation**

1. **Peak Detection**
Used signal peak detection to identify repetitions.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/medium%20dead.png)
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/heavy%20bench.png)

2. **Performance**
Achieved MAE of 1.02.
![image alt](https://github.com/Param10Patil/Intelligent-Fitness-Tracker-Analytics/blob/main/reports/figures/Evaluation.png)

## **Project Conclusion**
This pipeline successfully:
- Processes raw sensor data
- Extracts meaningful features
- Classifies exercises with ~99% accuracy
- Estimates repetitions reliably

---

## 📂 **Repository Structure**
```
AI Fitness Tracker Project
project-root/
│
├── data/                                 # Data storage folder
│   ├── final/                            # Final processed datasets ready for modeling
│   │   └── 03_data_features.pkl          # Final featured dataset
│   │
│   ├── interim/                          # Intermediate processed datasets
│   │   ├── 01_data_processed.pkl         # Initially cleaned dataset
│   │   ├── 02_outliers_removed_chauvenet.pkl  # Dataset after outlier removal
│   │   └── 03_data_features.pkl          # Dataset after feature engineering
│   │
│   └── raw/                              # Raw original data
│       └── MetaMotion.zip                # Raw zipped source dataset
│
├── reports/                              # Reports and generated outputs
│   ├── figures/                          # Visualizations and plots
│   └── placeholder                       
│
├── src/                                  # Source code of the project
│   ├── data/
│   │   └── make_dataset.py               # Script for loading and preparing dataset
│   │
│   ├── features/
│   │   ├── DataTransformation.py         # General data transformation utilities
│   │   ├── FrequencyAbstraction.py       # Frequency domain feature extraction
│   │   ├── TemporalAbstraction.py        # Time domain feature extraction
│   │   ├── build_features.py             # Main feature engineering pipeline
│   │   ├── count_repetitions.py          # Repetition counting logic
│   │   └── remove_outliers.py            # Outlier detection and removal
│   │
│   ├── models/
│   │   ├── LearningAlgorithms.py         # Machine learning algorithms implementation
│   │   └── train_model.py                # Model training script  
│   │
│   └── visualization/
│       └── visualize.py                  # Visualization and plotting script
│
└── README.md                             # Project overview and instructions
```

---
