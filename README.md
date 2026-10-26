# CrowdCast: Public Transport Crowding Prediction Using Machine Learning

> **Undergraduate Machine Learning-I Mini-Project**  
> Predicting passenger crowding levels on urban public transport buses at upcoming stops **before the bus arrives**.

---

## 1. Project Overview & Problem Statement

Urban commuters frequently experience unpredictable bus crowding: buses arrive overflowing during peak hours while adjacent departures may run half-empty. Existing municipal transit apps show bus arrival times via GPS countdowns but fail to inform waiting passengers **how crowded the bus will be** upon arrival.

**CrowdCast** formulates this challenge as an actionable, three-class supervised classification problem:
- **Low**: Bus occupancy below 50% of practical capacity (seats likely available).
- **Moderate**: Bus occupancy from 50% up to 79.9% (seating scarce, standing room available).
- **High**: Bus occupancy at or above 80% (approaching or exceeding crush capacity).

*Note: These thresholds are project-specific practical standards designed for urban transit commuter advisory, not universal legal regulations.*

---

## 2. Key Objectives & Learning Outcomes

1. **Pre-Arrival Future Prediction (Zero Target Leakage):** Formulate features using solely data known **before** reaching the destination stop (e.g., passenger count at previous stop, route, time of day, estimated travel minutes).
2. **Trip-Grouped Generalization:** Prevent data leakage across sequential stops by implementing **GroupShuffleSplit** and **GroupKFold** grouped on `trip_id`.
3. **Algorithm Comparison:** Train and compare an interpretable rule-based **Decision Tree Classifier** (primary) against an instance-based **K-Nearest Neighbors (KNN) Classifier** (comparison).
4. **End-to-End Pipeline:** Provide robust data collection, automated cleaning and validation audit, hyperparameter tuning, metric evaluation, and an interactive **Streamlit** user interface.

---

## 3. Dataset Schema & Data Dictionary

The survey data is recorded in `data/raw/crowdcast_data.csv` using the following schema:

| Column Name | Data Type | Description | Role in ML Pipeline |
|---|---|---|---|
| `observation_id` | String | Unique survey record ID (e.g. `OBS_0001`) | Metadata (Excluded from training) |
| `trip_id` | String | Bus trip identifier (e.g. `TRIP_20261026_01`) | Grouping Key (Train/Test split) |
| `observation_date` | String (YYYY-MM-DD) | Calendar date of survey | Temporal Metadata |
| `route_id` | Categorical | Bus route number/name (e.g. `Route_101`) | Input Feature (One-Hot Encoded) |
| `previous_stop` | Categorical | Stop where pre-prediction count occurred | Input Feature (One-Hot Encoded) |
| `target_stop` | Categorical | Upcoming stop where prediction applies | Input Feature (One-Hot Encoded) |
| `day_of_week` | Categorical | Monday through Sunday | Input Feature (One-Hot Encoded) |
| `prediction_time` | Time (HH:MM) | Time when pre-arrival observation was logged | Feature (Extracted Hour/Minute) |
| `target_observation_time` | Time (HH:MM) | Timestamp when bus actually arrived at target stop | Audit/Validation (Excluded from training) |
| `passengers_before_target`| Float | Observed passenger count at the previous stop | Input Numerical Feature |
| `practical_capacity` | Float | Verified bus capacity (seats + standees) | Input Numerical Feature |
| `minutes_since_previous_count` | Float | Estimated travel minutes to target stop | Input Numerical Feature |
| `is_peak_period` | Binary (0/1) | Rush hour indicator flag | Input Numerical Feature |
| `target_passenger_count` | Float | Actual ground-truth count at target stop | Audit / Target Source (**Excluded from training**) |
| `target_occupancy_percentage`| Float | `(target_passenger_count / practical_capacity) * 100` | Target Source (**Excluded from training**) |
| `crowding_level` | Categorical | Categorized class (`Low`, `Moderate`, `High`) | **Target Variable (`y`)** |

---

## 4. Leakage Prevention: Current-State vs. Future Prediction

```
+------------------------------------------------------------------------------------+
|               TIMELINE OF A BUS TRIP FROM PREVIOUS STOP TO TARGET STOP              |
+------------------------------------------------------------------------------------+
  [PREVIOUS STOP]                                                 [TARGET STOP]
         |                                                              |
         v                                                              v
  Time: t_previous                                               Time: t_target
  • passengers_before_target recorded                            • Bus reaches target stop
  • prediction_time recorded                                     • target_passenger_count observed
  • route_id, day_of_week, practical_capacity                    • target_occupancy_percentage calculated
  • minutes_since_previous_count estimated                       • crowding_level (Target Class) derived
         |                                                              |
         +------------------------ PREDICTION --------------------------+
         |                        MADE HERE                             |
         v                                                              v
  [INPUT FEATURES TO MODEL]                                      [SUPERVISED TARGET y]
  • Passengers at previous stop                                  • crowding_level (Low / Mod / High)
  • Earlier occupancy ratio                                      
  • Prediction hour and minute                                   *NEVER pass target counts,
  • Route, stops, day of week, capacity                          occupancy %, or target time
  • Minutes travel distance, peak flag                           into input feature matrix X!
```

- **Current-State Classification (Wrong formulation):** Predicting if the bus is currently crowded using sensors on the bus right now.
- **Future Crowding Prediction (CrowdCast formulation):** Predicting the crowding level at an upcoming stop **before** the bus arrives, enabling passengers at the platform to make boarding or alternative transit decisions.

---

## 5. Machine Learning Pipeline Flowchart

```
+----------------------------------------------------+
|            1. FIELD DATA COLLECTION                |
|  Manual survey via CLI (collect_data.py)           |
|  Appends raw records -> data/raw/crowdcast_data.csv|
+----------------------------------------------------+
                          |
                          v
+----------------------------------------------------+
|             2. PREPARATION & AUDIT                 |
|  Run src/prepare_data.py                           |
|  - Validates positive counts and capacities        |
|  - Checks timestamps (t_pred <= t_target)          |
|  - Derives target occupancy % and crowding classes |
|  - Drops duplicate IDs; writes clean CSV           |
+----------------------------------------------------+
                          |
                          v
+----------------------------------------------------+
|           3. GROUPED TRAIN / TEST SPLIT            |
|  GroupShuffleSplit on 'trip_id' (e.g. 75% / 25%)   |
|  Prevents trip data leakage across partitions      |
+----------------------------------------------------+
                          |
                          v
+----------------------------------------------------+
|            4. FEATURE PREPROCESSING                |
|  scikit-learn ColumnTransformer:                   |
|  - Numerical -> Median Imputer (+ StandardScaler)  |
|  - Categorical -> Most Frequent + OneHotEncoder    |
+----------------------------------------------------+
                          |
             +------------+------------+
             |                         |
             v                         v
+-------------------------+ +-------------------------+
|  5A. DECISION TREE      | |  5B. KNN CLASSIFIER     |
|  Primary Algorithm      | |  Comparison Algorithm   |
|  GroupKFold GridSearch  | |  GroupKFold GridSearch  |
|  (depth, split, leaf)   | |  (k, distance metric)   |
+-------------------------+ +-------------------------+
             |                         |
             +------------+------------+
                          |
                          v
+----------------------------------------------------+
|             6. MODEL SERIALIZATION                 |
|  Export joblib pipelines to models/                |
|  Export metadata & label encoder                   |
+----------------------------------------------------+
                          |
             +------------+------------+
             |                         |
             v                         v
+-------------------------+ +-------------------------+
|     7. EVALUATION       | |   8. WEB INTERFACE      |
|  Run evaluate.py        | |  Run streamlit run      |
|  - Confusion matrices   | |      app.py             |
|  - Class distribution   | |  - Interactive inputs   |
|  - Decision tree visual | |  - Confidence bar charts|
|  - Macro F1 & Recall    | |  - Model comparisons    |
+-------------------------+ +-------------------------+
```

---

## 6. Project Directory Structure

```
CrowdCast/
│
├── data/
│   ├── raw/
│   │   └── crowdcast_data.csv          # Raw survey observations CSV
│   └── processed/
│       └── crowdcast_clean.csv         # Validated and prepared clean dataset
│
├── models/
│   ├── decision_tree_model.joblib      # Tuned Decision Tree scikit-learn Pipeline
│   ├── knn_model.joblib                # Tuned KNN scikit-learn Pipeline
│   ├── label_encoder.joblib            # Target LabelEncoder
│   └── model_metadata.json             # Feature lists, best params, test metrics
│
├── reports/
│   ├── holdout_test_predictions.csv    # Predictions on unseen test trips
│   ├── training_set.csv                # Training split observations
│   ├── confusion_matrices.png          # Side-by-side confusion matrix plot
│   ├── class_distribution.png          # Train vs Test class balance plot
│   └── decision_tree_structure.png     # Visual tree rule splits diagram
│
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── prepare_data.py                 # Data validation and target derivation script
│   └── train.py                        # Trip-grouped training and hyperparameter tuning
│
├── collect_data.py                     # CLI assistant for recording survey observations
├── evaluate.py                         # Evaluation metrics, diagnostics, and plots
├── app.py                              # Streamlit web prediction application
├── requirements.txt                    # Project Python dependencies
├── README.md                           # Comprehensive documentation and project guide
└── .gitignore                          # Standard git ignore definitions
```

---

## 7. Step-by-Step Installation & Execution Guide

### Step 1: Open Terminal in Project Root
Open PowerShell in the `CrowdCast` folder:
```powershell
Set-Location -Path "d:\ML\CrowdCast"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Record Survey Observations
To record field observations interactively:
```powershell
python collect_data.py
```
*(For testing software workflows when physical field surveys are in progress, generate a test scaffold using: `python collect_data.py --demo-synthetic --count 400`)*

### Step 4: Validate and Clean Dataset
```powershell
python src/prepare_data.py
```
This runs the audit checks, enforces timestamp chronology, checks for non-negative passengers, derives occupancy percentages, and creates `data/processed/crowdcast_clean.csv`.

### Step 5: Train and Tune Models
```powershell
python src/train.py
```
Executes trip-grouped train/test splitting, builds preprocessing pipelines, performs `GroupKFold` cross-validation hyperparameter tuning, and exports serialised models to `models/`.

### Step 6: Evaluate Performance and Generate Plots
```powershell
python evaluate.py
```
Outputs classification reports, macro F1, and saves diagnostic figures to `reports/`.

### Step 7: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test predictions interactively.

---

## 8. Measured Performance & Experimental Results

*Measured on 400 observations across 100 trips (Holdout Test: 100 observations across 25 unseen trips).*

### Benchmark Summary Table

| Metric | Decision Tree (Primary) | K-Nearest Neighbors (KNN) | Academic Observation |
|---|---|---|---|
| **Overall Accuracy** | **87.00%** | 72.00% | Decision Tree is significantly higher |
| **Macro Precision** | **0.8604** | 0.6943 | DT generates far fewer false positives |
| **Macro Recall** | **0.8693** | 0.6944 | DT captures minority classes more reliably |
| **Macro F1-Score** | **0.8608** | 0.6911 | Balanced multi-class metric shows clear DT advantage |
| **Tuned Parameters** | `max_depth=3`, `criterion='entropy'` | `k=5`, `metric='manhattan'`, `weights='distance'` | Shallow tree prevents overfitting |

### Per-Class Detailed Breakdown

```
Decision Tree Classifier:
              precision    recall  f1-score   support
         Low     0.8235    1.0000    0.9032        28
    Moderate     0.8077    0.7241    0.7636        29
        High     0.9500    0.8837    0.9157        43

KNN Classifier:
              precision    recall  f1-score   support
         Low     0.7857    0.7857    0.7857        28
    Moderate     0.5217    0.4138    0.4615        29
        High     0.7755    0.8837    0.8261        43
```

### Why Decision Tree Outperformed KNN:
1. **Curse of Dimensionality with One-Hot Encoded Stops:** KNN computes geometric distances. When categorical stops and routes are one-hot encoded into 15+ dimensions, all points tend to become equidistant in sparse binary space, degrading KNN's distance metric.
2. **Orthogonal Decision Boundaries:** Decision Trees isolate critical continuous ratios (such as `earlier_occupancy_ratio >= 0.72`) with simple axis-parallel cuts, ignoring noisy or uninformative stop indicators.

---

## 9. Academic Project Report Outline

For undergraduate submissions, structure your mini-project report as follows:

1. **Title Page & Certificates:** Project title, candidate names, supervisor details, institution.
2. **Abstract:** 200–250 words summarizing the problem, dataset, Decision Tree vs. KNN approach, and key test results.
3. **Introduction:** Urban public transit challenges, passenger comfort, motivation for crowding prediction.
4. **Problem Formulation:** Definition of pre-arrival forecasting vs. current-state classification; mathematical formulation of `occupancy_percentage`.
5. **Literature Review:** Prior work on smart transit systems, automated passenger counting (APC), and baseline ML models.
6. **Dataset Collection & Schema:** Data collection methodology, survey sheet design, schema dictionary, validation constraints.
7. **Leakage Prevention & Grouped Splitting:** Explanation of chronological leakage, target leakage, and trip-grouped cross-validation.
8. **Methodology:** Preprocessing pipeline (imputation, scaling, one-hot encoding), Decision Tree theory, KNN distance theory.
9. **Experimental Results & Discussion:** Accuracy, Macro F1, confusion matrices, per-class breakdown, why algorithms differed.
10. **Web Application Implementation:** Streamlit UI design, validation safeguards, real-time inference flow.
11. **Conclusion & Future Scope:** Summary of findings, potential integration of live automated APC sensor hardware, weather API feeds.
12. **References:** Standard IEEE or APA citations.

---

## 10. Viva-Voce Questions & Comprehensive Answers

### Q1: What is the core objective of CrowdCast?
**Answer:** CrowdCast predicts the crowding category (`Low`, `Moderate`, or `High`) of a transit bus at an upcoming stop **before** the bus arrives, enabling commuters to plan boarding or alternate routes.

### Q2: Why is this framed as a classification problem rather than a regression problem?
**Answer:** While exact passenger counts can be continuous, commuters need actionable qualitative advice ("Will I get a seat?" or "Is the bus packed?"). Small count errors (e.g., 42 vs 45 passengers) do not alter commuter decisions, making 3-class classification more reliable and practical.

### Q3: How is the target variable `crowding_level` derived?
**Answer:** It is derived from the target occupancy percentage:  
$$\text{occupancy\_percentage} = \left(\frac{\text{target\_passenger\_count}}{\text{practical\_capacity}}\right) \times 100$$  
Thresholds: `< 50%` = Low, `50% - 79.9%` = Moderate, `≥ 80%` = High.

### Q4: What is data leakage, and how did you prevent it in this project?
**Answer:** Data leakage occurs when information from the future or target is included in input features during training. We strictly excluded `target_passenger_count`, `target_occupancy_percentage`, and `target_observation_time` from model inputs. Only features observed before the bus arrives at the target stop are used.

### Q5: Why can't we use a standard random `train_test_split`?
**Answer:** A bus trip has sequential stops. If observations from the same trip are randomly split between training and testing, the model can memorize the trip's specific day, weather, or driver pattern, leading to overly optimistic "data leakage" across stops. We used `GroupShuffleSplit` on `trip_id` so entire trips remain unseen in the test set.

### Q6: Why did you choose Decision Tree as your primary algorithm?
**Answer:** Decision Trees produce human-interpretable "if-then" rules that match how transit inspectors think (e.g., "If occupancy at previous stop > 70% and peak hour, then High crowding"). They handle mixed categorical and numerical data without requiring feature scaling.

### Q7: Why did KNN require feature scaling (`StandardScaler`), whereas Decision Tree did not?
**Answer:** KNN calculates Euclidean or Manhattan distances between vectors. Features with large numerical ranges (e.g., capacity = 60) would dominate smaller features (e.g., peak period = 1). Decision Trees only evaluate threshold splits on one feature at a time, making them invariant to monotonic scaling.

### Q8: What distance metric worked best for KNN, and why?
**Answer:** Manhattan distance (`p=1`) performed better than Euclidean distance (`p=2`) because in datasets with one-hot encoded binary features, Manhattan distance is less prone to the distortions caused by high-dimensional sparsity.

### Q9: Why is Macro F1-Score more informative than Accuracy for this project?
**Answer:** Transit data often exhibits class imbalance (e.g., fewer "Moderate" observations compared to "High" rush-hour or "Low" off-peak observations). Overall accuracy can be misleadingly high if a model only predicts the majority class. Macro F1 averages F1 across all classes with equal weight, penalizing models that fail on minority classes.

### Q10: What is the practical capacity of a bus, and why does it matter?
**Answer:** Practical capacity is the verified count of all seats plus safe allowable standing passengers (typically 50–70 on standard city buses). Using seated-only capacity would falsely classify normal standing loads as 150% crowded.

### Q11: What hyperparameter tuning was performed on the Decision Tree?
**Answer:** We tuned `max_depth` (limiting tree depth to prevent overfitting), `min_samples_split`, `min_samples_leaf`, and split criterion (`gini` vs `entropy`) using 5-fold `GroupKFold` cross-validation.

### Q12: Why did "Moderate" crowding have lower recall than "Low" or "High"?
**Answer:** "Moderate" occupies the intermediate boundary (50%–79%). If a bus with 60 capacity gains 3 passengers, occupancy shifts from 48% (Low) to 53% (Moderate). These boundary fluctuations make intermediate states inherently harder to distinguish than clear extremes.

### Q13: What role does `ColumnTransformer` play in your scikit-learn pipeline?
**Answer:** `ColumnTransformer` applies different transformations to different subsets of columns: median imputation and standard scaling to numerical columns, while applying most-frequent imputation and one-hot encoding to categorical stop/route columns, preventing data snooping during CV.

### Q14: How does `app.py` ensure safe predictions?
**Answer:** It performs input validation checking that passenger counts cannot be negative and alerts users if passenger count exceeds 1.5× capacity. It passes inputs through the exact saved preprocessor pipeline without hand-coded feature mismatches.

### Q15: What are the primary limitations of this dataset and project?
**Answer:** 
1. Small sample size collected over a limited survey window.
2. Sudden external shocks (heavy rain, transit strikes, traffic accidents) are not captured in static schedule features.
3. Model is route-specific and cannot generalize to unobserved routes without retraining.

---

## 11. Practical Data Collection Guidance for Students

### Example Observation Survey Sheet

Print or copy this template when conducting field counts at transit stops:

| Obs ID | Trip ID | Date | Route | Prev Stop | Target Stop | Pred Time | Target Time | Earlier Pax | Bus Cap | Travel Mins | Peak? | Target Pax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OBS_001 | TRIP_01 | 2026-10-27 | 101 | Station | Square | 08:30 | 08:38 | 35 | 60 | 8.0 | Yes | 52 |
| OBS_002 | TRIP_01 | 2026-10-27 | 101 | Square | TechPark | 08:40 | 08:47 | 48 | 60 | 7.0 | Yes | 55 |

### Collection Rules:
1. **Count at Previous Stop:** Stand at the previous stop or ride the bus to record `passengers_before_target`.
2. **Note the Time:** Log the exact departure time (`prediction_time`).
3. **Record Arrival at Target Stop:** Note the count on board after boardings/alightings at the target stop (`target_passenger_count`).
4. **Target Size:** Strive for 300–500 observations across different days and hours to ensure balanced representation of Low, Moderate, and High classes.
