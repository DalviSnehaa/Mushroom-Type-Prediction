# Mushroom-Type-Prediction
The goal is to develop a machine learning model that predicts whether a mushroom is poisonous or edible based on its physical and habitat characteristics. This project can aid in identifying potentially harmful mushrooms and enhance safety during foraging.

## Objective
Predict whether a mushroom is **poisonous** or **edible** using its physical and environmental attributes. The model leverages machine learning algorithms to achieve high accuracy.

## Dataset Overview
- **Features:** 22 attributes, including `cap-shape`, `cap-color`, `odor`, `gill-color`, `stalk-shape`, `population`, and `habitat`.
- **Target:** `class` (Edible: 0, Poisonous: 1).

## Problem Statement
1. Use **Decision Tree (DT)** to classify mushrooms as poisonous or edible.
2. Evaluate **Logistic Regression** as an alternative model.
3. Optimize the DT model by selecting important features based on feature importance.

## Workflow
1. **Preprocessing:** Encode categorical variables and clean data.
2. **Feature Engineering:** Use feature importance to select key predictors.
3. **Model Training and Evaluation:**
   - Logistic Regression:
     - Recall = 93%, Precision = 94%, Accuracy = 93%.
   - Decision Tree:
     - Recall = 93%, Precision = 95%, Accuracy = 93% (all features).
     - Accuracy = 100% (important features only).


