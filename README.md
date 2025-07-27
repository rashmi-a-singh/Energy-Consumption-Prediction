# Predictive Modeling for Building Energy Consumption

Developed a predictive model to forecast hourly building energy usage based on historical weather patterns and building characteristics. This project serves as an end-to-end demonstration of a typical time-series forecasting problem in data science.

![Project Visualization](https://i.imgur.com/gKzT4oV.png) 
*An example visualization comparing the model's predictions (red) against actual energy usage (blue).*

---

## 📋 Table of Contents
- [Project Goal](#-project-goal)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [Data Source](#-data-source)
- [How to Run This Project](#-how-to-run-this-project)
- [Project Walkthrough](#-project-walkthrough)
- [What I Learned](#-what-i-learned)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Goal

The primary objective of this project was to build a machine learning model that accurately predicts the energy consumption (`meter_reading`) of a building. This involved cleaning and preparing complex time-series data, engineering relevant features, and training a robust regression model.

---

## ✨ Key Features

- **Data Integration:** Merged three separate datasets (building energy readings, building metadata, and weather data) into a single, analysis-ready master table.
- **Data Cleaning:** Handled missing values through interpolation to ensure data quality and model stability.
- **Feature Engineering:** Extracted valuable time-based features from timestamps (e.g., hour, day of the week, month) to help the model capture cyclical energy patterns.
- **Model Training:** Implemented an **XGBoost Regressor**, a powerful gradient-boosting algorithm known for its performance and accuracy.
- **Model Evaluation:** Assessed the model's performance using Root Mean Squared Error (RMSE) and visualized the predictions against actual values to qualitatively judge its effectiveness.

---

## 🛠️ Technologies Used

- **Python:** The core programming language for the project.
- **Pandas:** Used for all data manipulation, including loading, merging, and cleaning the datasets.
- **XGBoost:** The machine learning library used to build the predictive regression model.
- **Scikit-learn:** Utilized for splitting the data into training and testing sets and for model evaluation metrics.
- **Matplotlib:** Used for creating the final visualizations to compare predicted vs. actual results.
- **Jupyter / Google Colab:** The development environment used for interactive coding and analysis.

---

## 📊 Data Source

The data for this project is from the **ASHRAE - Great Energy Predictor III** competition on Kaggle. The dataset includes over 20 million rows of energy readings from more than 1,400 buildings over a one-year period.

Due to its large size, the data is not included in this repository. You can download it directly from the source:
[https://www.kaggle.com/c/ashrae-energy-prediction/data](https://www.kaggle.com/c/ashrae-energy-prediction/data)

---

## 🚀 How to Run This Project

To replicate this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Install the necessary libraries:**
    ```bash
    pip install pandas xgboost scikit-learn matplotlib
    ```
3.  **Download the data** from the Kaggle link provided above. Place the following files in the root directory of the project:
    - `train.csv`
    - `building_metadata.csv`
    - `weather_train.csv`
4.  **Run the Python script:**
    ```bash
    python your_script_name.py
    ```

---

## 🚶‍♂️ Project Walkthrough

The project was structured in a step-by-step manner to ensure clarity and logical progression:

1.  **Data Loading:** The three separate CSV files were loaded into pandas DataFrames.
2.  **Data Merging:** The DataFrames were merged based on common keys (`building_id`, `site_id`, `timestamp`) to create one comprehensive dataset.
3.  **Data Filtering & Cleaning:** To make the analysis manageable, the dataset was filtered for a single building. Missing values, particularly in the weather data, were filled using linear interpolation.
4.  **Feature Engineering:** The `timestamp` column was broken down into more machine-learning-friendly features, such as `hour` and `dayofweek`, which are strong indicators of energy consumption patterns.
5.  **Model Training & Prediction:** The data was split into a training set (the first 80% of the data) and a testing set (the final 20%). An XGBoost model was trained on the training set and then used to make predictions on the testing set.
6.  **Visualization:** The model's predictions were plotted against the actual `meter_reading` values from the test set to visually assess its performance.

---

## 🧠 What I Learned

This project was a foundational learning experience. Key takeaways include:

- **The Critical Role of Data Preparation:** I learned that a significant portion of a data science project is dedicated to cleaning, merging, and preparing data. The quality of the final model is directly dependent on the quality of the input data.
- **The Power of Feature Engineering:** Simply feeding a model raw data is not enough. Creating insightful features, like extracting the hour from a timestamp, provides the model with the cyclical context needed to make accurate time-series predictions.
- **Practical Model Implementation:** I gained hands-on experience training an industry-standard model like XGBoost and understanding its core mechanics, from training to prediction and evaluation.
- **The Importance of Iteration:** My first model run (on Building 0) produced a correct but uninteresting flat line. This taught me that data science is an iterative process of testing, observing, and refining your approach.

---

## 🔮 Future Improvements

- **Hyperparameter Tuning:** Use techniques like Grid Search or Randomized Search to find the optimal parameters for the XGBoost model, potentially improving its accuracy.
- **Try Different Models:** Implement and compare the performance of other time-series models, such as Facebook's **Prophet** or a deep learning model like an **LSTM**.
- **Scale the Analysis:** Expand the project to train models for all buildings in the dataset, perhaps developing a single, more generalized model or one model per building type (`primary_use`).


