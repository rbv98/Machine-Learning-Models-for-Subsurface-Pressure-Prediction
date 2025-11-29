📊 Machine Learning Models for Subsurface Pressure Prediction: A Data Mining Approach
This repository contains the implementation of a novel Hybrid Meta-Ensemble (HME) framework for pore pressure prediction in oil and gas drilling operations, as described in our paper published in Computers (2025).

🎯 Overview
Accurate pore pressure prediction is critical for safe and efficient drilling operations. This project introduces a comprehensive machine learning framework that combines five diverse models (CNN, RNN, DFNN, Random Forest, and XGBoost) through an advanced meta-ensemble approach to achieve superior prediction accuracy (R² = 0.938) on blind well validation.


📁 Repository Structure
The repository is organized with separate directories for data (well log CSV files), saved models, generated figures, and the main Jupyter notebook that implements the entire workflow. The dataset includes four wells from the Potwar Basin in Pakistan, with three wells used for training and one held out as a completely blind test well.

🔬 Methodology Workflow

Phase 1: Data Collection and Initial Processing

The workflow begins by loading well log data from four different wells in the Missa Keswal field of the Potwar Basin. Each well contains thousands of depth-indexed measurements spanning approximately 1000 to 3000 meters depth. The data includes various geophysical measurements such as gamma ray readings (which indicate clay content and lithology), sonic transit times (reflecting rock compaction), bulk density measurements, deep resistivity readings, and computed pressure profiles.
The preprocessing stage handles missing values, standardizes column naming conventions, and removes invalid readings marked with placeholder values. Feature selection is performed based on geophysical importance, retaining eight key parameters that are most relevant for pressure prediction. The target variable, pore pressure, is computed using Eaton's empirical method and calibrated with available measured pressure data where possible.

Phase 2: Addressing Geological Heterogeneity

One of the major challenges in cross-well pressure prediction is the significant geological variation between different wells. Statistical analysis reveals substantial distributional differences - for instance, average pore pressure varies from 2822 psi in training wells to 2289 psi in the blind test well. To address this challenge, the framework implements distribution alignment using quantile transformation methods.
This alignment process transforms the feature distributions to follow a standard normal distribution, ensuring that all features contribute equally to model training regardless of their original measurement scales. This is particularly important given the diverse units and ranges of well log measurements, from API units for gamma ray to ohm-meters for resistivity.

Phase 3: Base Model Development

The framework employs five distinct machine learning models, each capturing different aspects of the complex relationships in well log data:

Deep Learning Models:
The Convolutional Neural Network uses one-dimensional convolutions to identify local patterns in the well log signatures, similar to how it might detect edges in images but adapted for sequential well data
The Recurrent Neural Network with LSTM cells captures sequential dependencies as depth increases, understanding how measurements at one depth relate to those above and below
The Deep Feedforward Neural Network serves as a universal function approximator, learning complex nonlinear mappings between input features and pressure

Tree-Based Ensemble Models:
Random Forest employs 500 decision trees with bootstrap aggregation, providing stability through ensemble voting and handling the high variability common in geological data
XGBoost uses gradient boosting with 1000 sequential trees, each correcting the errors of previous trees, making it particularly effective for capturing complex nonlinear relationships

Each model is trained independently on the same preprocessed dataset, with careful attention to hyperparameter tuning and regularization to prevent overfitting.

Phase 4: Initial Blind Well Testing

When the trained models are directly applied to the completely unseen blind well (MISSA KESWAL-02), the results are surprisingly poor - all models produce negative R² values, indicating predictions worse than simply using the mean pressure value. This dramatic failure highlights the severe domain shift between training and test wells due to different depositional environments, compaction histories, and pressure generation mechanisms.

Phase 5: Domain Adaptation Through Strategic Fine-Tuning

To bridge the geological gap between training and test wells, the framework implements a controlled fine-tuning strategy. Fifteen percent of the blind well data (approximately 785 samples) is strategically incorporated into the training process. This small amount of target domain data enables the models to recalibrate for the new geological regime without overfitting to the limited samples.
The fine-tuning process differs for each model type. Neural networks undergo additional training epochs with frozen early layers to preserve general feature extraction while adapting final layers. Tree-based models are retrained from scratch with the augmented dataset, as their structure doesn't support layer-wise fine-tuning.

Phase 6: Meta-Ensemble Construction

The key innovation lies in the meta-ensemble approach. Rather than simply averaging model predictions, a meta-learner (Ridge regression model) is trained on the stacked predictions from all five base models. This meta-learner learns optimal, nonlinear combinations of base model outputs based on the geological context.
The stacking process works by treating each base model's prediction as a new feature. The meta-learner then discovers which models are most reliable under different conditions. For instance, it might weight XGBoost more heavily in normally compacted zones while relying more on neural networks in complex transition zones.

Phase 7: Performance Evaluation

The evaluation employs multiple metrics to assess different aspects of prediction quality:

R² score measures overall variance explained by the model
Root Mean Square Error (RMSE) penalizes large errors more heavily, important for safety-critical applications
Mean Absolute Error (MAE) provides an intuitive measure of typical prediction deviation
Relative RMSE normalizes errors as a percentage of the pressure range for cross-well comparison

Phase 8: Cross-Validation Analysis

To ensure robust performance assessment, Leave-One-Well-Out Cross-Validation (LOWO-CV) is performed. Each of the four wells serves as a test well in turn, with the remaining three used for training. This approach is more rigorous than random splitting as it ensures the test well represents a completely different geological setting, providing realistic assessment of operational deployment scenarios.

Phase 9: Visualization and Interpretation

The workflow generates comprehensive visualizations to understand model behavior and performance:
Well Log Analysis creates five-track composite displays showing the relationship between input features and pressure profiles. These plots use color gradients to represent depth progression and highlight overpressured zones where pore pressure exceeds hydrostatic pressure.
Performance Comparisons visualize how each model performs across different metrics, with bar charts showing R² scores, RMSE values, and MAE for easy comparison. Box plots reveal the distribution of predictions and identify outliers.
Scatter Plots display actual versus predicted pressures for each model, with perfect predictions falling along the diagonal line. The spread around this line indicates prediction uncertainty.
Residual Analysis examines prediction errors across the pressure range, revealing whether models exhibit systematic biases at certain pressure levels or depth intervals.
Distribution Analysis shows how well the distribution alignment process worked, comparing feature distributions between training and test wells before and after transformation.

Phase 10: Data Injection Experiments

Additional experiments explore the minimum amount of target well data required for effective adaptation. Tests with 10% and 25% data injection reveal that even 10% (519 samples) provides substantial improvement, though 15% offers the best balance between performance and data efficiency.

🎯 Key Insights from the Workflow

The workflow demonstrates that successful cross-well pressure prediction requires more than just sophisticated algorithms. The dramatic initial failure (negative R² values) followed by successful adaptation illustrates the importance of domain adaptation in geological machine learning. The meta-ensemble approach proves superior to individual models by learning complementary strengths - neural networks capture continuous patterns while tree-based models handle discrete geological boundaries effectively.
The framework's success with minimal target well data (15%) makes it practically viable for real-world deployment where extensive labeled data from new wells is rarely available. The systematic workflow from data processing through ensemble construction provides a template applicable to other geoscience prediction challenges beyond pore pressure.

📈 Performance Summary
After complete execution of the workflow:

Individual models achieve R² values ranging from 0.77 to 0.94 after fine-tuning.
The meta-ensemble reaches R²=0.938 with RMSE of 201 psi.
Cross-validation confirms robust performance with mean R²=0.959 (±0.031).
The framework successfully handles pressure regime variations of over 500 psi between wells.
