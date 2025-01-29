import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset (CSV should be in the same directory as this script)
df = pd.read_csv('emotion_data.csv')

# Splitting into features and labels  
X = df.drop(columns='label').values  # Using all columns except 'label' as features  
y = df['label'].values  # Target labels (emotions)

# Splitting the data into training (80%) and testing (20%) sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features (neural networks perform better when data is scaled)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining the neural network  
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),  # Input layer  
    Dropout(0.2),  # Dropout to prevent overfitting  
    Dense(64, activation='relu'),  # Hidden layer  
    Dropout(0.2),  # More dropout for regularization  
    Dense(4, activation='softmax')  # Output layer (assuming 4 emotion classes)
])

# Compiling the model with Adam optimizer and categorical cross-entropy loss  
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model (50 epochs, batch size of 32)  
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluating the model on the test set  
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Saving the trained model for later use  
model.save('emotion_recognition_model.keras')

print("Model training complete and saved successfully!")
