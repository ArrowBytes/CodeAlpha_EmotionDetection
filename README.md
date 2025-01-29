# CodeAlpha_EmotionDetection
This project focuses on classifying emotions from speech data using a neural network. The model is trained on processed audio features extracted from the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song). It predicts emotions based on numerical representations of speech recordings.


Source: The dataset used for this project is the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).
Processing:
Only the speech and song audio files were used, excluding video data.
Extracted numerical features from the audio using MFCCs (Mel-Frequency Cepstral Coefficients).
Stored the processed features and corresponding labels in emotion_data.csv.

Feature Extraction: Converted audio signals into meaningful numerical representations (MFCCs).
Label Encoding: Categorical emotion labels were converted into integers for model compatibility.
Data Scaling: Standardized the extracted features using StandardScaler to improve model performance.
Data Splitting:
80% Training Data
20% Test Data

Model Archictecture
The model is a fully connected neural network (MLP) implemented using TensorFlow/Keras.

Input Layer: Accepts the preprocessed audio features.
Hidden Layers:
Dense layers with ReLU activation for learning representations.
Dropout layers to prevent overfitting.
Output Layer: Uses a softmax activation function to classify the emotions.



Key Libraries Used:
pandas – Data handling
numpy – Numerical operations
scikit-learn – Data preprocessing & evaluation
tensorflow – Deep learning framework
matplotlib – Visualization

Achieved a test accuracy of 77.78% 


RAVDESS Dataset: https://zenodo.org/record/1188976
TensorFlow Documentation: https://www.tensorflow.org/
