# Machine Learning for Early Disease Detection

I worked with precision medicine company Simpatica Medicine to create ensemble models which predict the presence of a neurodegenerative disease in a patient and important genes for disease detection.

The original dataset contained 60k features; I compared Random Forest and Lasso models as novel forms of dimensionality reduction to identify predictive features. I then ran the identified features through Gradient Boosted models to predict if a patient has the disease and find the most important features for prediction. My final model resulted in a F1 score of 96% (baseline: 91%).

**NOTE:**
Due to privacy concerns, the nature of the genomic data and the specific disease that I was working with will not be revealed. Therefore,  my files, including EDA and data transformation, have been updated to remove confidential information.
