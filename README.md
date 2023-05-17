#  Popularity Score of Music Tracks

## Team Members

- Giacomo Collesei 770271
- Spencer Patrick Gengis Marcinko 771411
- Fatma Soussi 770001

## Introduction
This project focuses on predicting the popularity of music tracks based on various features. The goal is to develop a machine learning system that can accurately estimate the popularity of a track given its attributes such as track genre, artist information, and other related factors. By analyzing and understanding the factors that contribute to a track's popularity, we aim to provide insights for music industry professionals and enthusiasts.

## Dataset Description
The dataset contains the following variables:
- track_id: A unique identifier for each track.
- artists: The artists who performed the track. A single track can have multiple artists, separated by a comma.
- album_name: The name of the album that the track appears on.
- track_name: The name of the track.
- popularity: The popularity score of the track, ranging from 0 to 100.
- duration_ms: The duration of the track in milliseconds.
- explicit: A binary value indicating whether the track contains explicit lyrics.
- danceability: A score indicating how danceable the track is, ranging from 0 to 1.
- energy: A score indicating the energy level of the track, ranging from 0 to 1.
- key: The key that the track is in (e.g., C, D, E, etc.).
- loudness: The loudness of the track in decibels (dB).
- mode: The mode of the track (major or minor).
- speechiness: A score indicating how much speech-like content is in the track, ranging from 0 to 1.
- acousticness: A score indicating how acoustic the track is, ranging from 0 to 1.
- instrumentalness: A score indicating how instrumental the track is, ranging from 0 to 1.
- liveness: A score indicating the presence of an audience in the recording, ranging from 0 to 1.
- valence: A score indicating the positivity of the track, ranging from 0 to 1.
- tempo: The tempo of the track in beats per minute (BPM).
- time_signature: The time signature of the track (e.g., 4/4, 3/4, etc.).
- track_genre: The genre of the track (if available).

## Methods
In this project, we employed a combination of data preprocessing, feature engineering, and machine learning techniques to predict track popularity. The following steps were taken:

1. Data Exploration: We analyzed the provided dataset to understand its structure, variables, and potential correlations. We observed that the dataset contains information about music tracks, including track names, genres, artists, and popularity ratings.
![correlation_before_merge](https://github.com/giacolle6/770271/assets/127773034/08192f87-622c-443b-a35d-609c94353294)
![correlation_after_merge](https://github.com/giacolle6/770271/assets/127773034/396fe8f5-b608-4d44-959e-601b46f1f153)
![correlation_heatmap](https://github.com/giacolle6/770271/assets/127773034/0a62034b-76a6-46bb-8187-06042aaf939b)

2. Data Cleaning: We performed data cleaning to ensure the integrity and quality of the dataset. We removed any missing or irrelevant data points and handled duplicates. By cleaning the data, we ensured that our models are trained on reliable and accurate information.

3. Feature Engineering: To enhance the predictive power of our models, we created additional features based on domain knowledge and statistical analysis. For example, we converted the track popularity variable into a binary variable, indicating whether a track is popular or not. We also derived artist popularity bins by grouping artists based on their average popularity ratings.

4. Encoding: We employed various encoding techniques to represent categorical variables, such as track genre and artist name, in a suitable numerical format for machine learning algorithms. For track genre, we utilized one-hot encoding to convert the categorical variable into binary features, enabling our models to effectively leverage this information. Additionally, we performed a grouping operation on artist names based on their average popularity. This allowed us to categorize artists into four distinct groups: high, medium, low, and poor, providing a coherent representation of their popularity levels within the dataset.
![genre_histogram](https://github.com/giacolle6/770271/assets/127773034/5a0820b8-3361-4381-b34d-2669f411c522)

![artist_popularity_histogram](https://github.com/giacolle6/770271/assets/127773034/fb5ede2c-bb6b-4bdb-9eb4-22c1377ae32e)

5. Feature Scaling: To ensure unbiased influence during the model training process, we performed feature scaling on the numerical features. We applied standardization to scale the numerical features to a common range, enabling fair comparisons and preventing any single feature from dominating the model.

6. Model Training: We selected a neural network model as our primary approach for music popularity prediction. The model consisted of multiple hidden layers, allowing it to capture complex relationships between the input features and the target variable. We used the Keras library to build and train the neural network model.

7. Cross-Validation: To evaluate the performance of our model and mitigate overfitting issues, we employed k-fold cross-validation. This technique involves splitting the dataset into k subsets, training the model on k-1 subsets, and evaluating its performance on the remaining subset. We used the mean squared error (MSE) as our evaluation metric to assess the prediction accuracy of our model.

### Environment Setup
To recreate the environment for this project, please follow these steps:

1. Install the required libraries specified in the `requirements.txt` file.
2. Ensure that the dataset file, `popularity_score_dataset.csv`, is available in the project directory.
3. Run the code in the provided Python script, ensuring that the necessary dependencies and environment settings are met.

### Flowchart

Start --> DataCleaning[Data Cleaning]
RemoveDuplicates[Remove duplicates] --> DataCleaning
HandleMissingValues[Handle missing values] --> DataCleaning
FeatureEngineering[Feature Engineering] --> DataCleaning
ExtractFeatures[Extract useful features] --> FeatureEngineering
CreateNewFeatures[Create new features] --> FeatureEngineering
DataEncoding[Data Encoding] --> FeatureEngineering
OneHotEncoding[Perform one-hot encoding] --> DataEncoding
StandardizeNumericalFeatures[Standardize numerical features] --> DataEncoding
ModelTraining[Model Training] --> DataEncoding
NeuralNetworkModel[Utilize a neural network model] --> ModelTraining
KFoldCrossValidation[Implement k-fold cross-validation] --> ModelTraining
ModelTuning[Model Tuning] --> ModelTraining
Hyperparameters[Experiment with different hyperparameters] --> ModelTuning
ModelArchitectures[Explore different model architectures] --> ModelTuning
ModelEvaluation[Model Evaluation] --> ModelTuning
MSE[Calculate Mean Squared Error (MSE)] --> ModelEvaluation

## Experimental Design

We conducted several experiments to validate the target contribution(s) of the project. The main purpose of each experiment is as follows:

### Experiment 1: 
To evaluate the performance of our baseline model, we trained a linear regression model using scikit-learn and compared its performance to our neural network model.

### Experiment 2: 
To evaluate the impact of feature engineering, we trained our model on two different versions of the dataset: one with only the original features and another with the additional features we engineered.

For both experiments, we used the MSE and R2 score as our evaluation metrics.

## Results
Our final model achieved an average MSE of 119.58 and an R2 score of 0.59 across all folds of the k-fold cross-validation. This indicates that our model is able to predict the popularity of tracks with moderate accuracy. Our experiments showed that feature engineering significantly improved the performance of the model compared to the baseline.

![cross_validation_results](https://github.com/giacolle6/770271/assets/127773034/dc4f60fd-1cbb-4042-8113-1914d7fe4550)


## Conclusions
In conclusion, our project aimed to predict the popularity of songs on Spotify using a neural network model. We began by cleaning the data, handling missing values and removing duplicates. Then, we engineered features such as genre and average artist popularity. We encoded the data and trained the model using Keras and TensorFlow. Our evaluation metrics indicated that the model performed well. The mean squared error and R2 score were both low, indicating that the model accurately predicted song popularity.

However, our work leaves some questions unanswered, such as whether the model would perform well on a different dataset or with different features. Future work could explore these questions and also investigate the impact of other factors, such as lyrics and album art, on song popularity. Overall, our project highlights the potential of using machine learning to predict song popularity and paves the way for further research in this area.

