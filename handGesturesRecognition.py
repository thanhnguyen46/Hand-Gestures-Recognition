# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
tf.random.set_seed(3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, Input

# HANDLING DATASET

# Set the dataset directory
dataset_dir = 'dataset'

# Generate folder names for each class (0-9)
folders_names = []
for i in range(10):
    folder = os.path.join(dataset_dir, f'0{i}')
    folders_names.append(folder)

# Define file names for each gesture
files_names = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

# Initialize an empty training data list
training_data = []

# Create training data
def create_training_data():
    for folder in folders_names:
        Class_num = folder[-1]
        print('Class', Class_num)
        for file in files_names:
            path = os.path.join(folder, file)
            print(file)
            for img in tqdm(os.listdir(path)):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_array, int(Class_num)])

# Call training data function
create_training_data()

# Shuffle the training data
random.shuffle(training_data)

# Preprocess the data
X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split the data into training, validation, and testing sets
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

train_size = int(len(X) * train_ratio)
validation_size = int(len(X) * validation_ratio)
test_size = len(X) - train_size - validation_size

X_train = X[:train_size]
y_train = y[:train_size]
X_valid = X[train_size:train_size+validation_size]
y_valid = y[train_size:train_size+validation_size]
X_test = X[train_size+validation_size:]
y_test = y[train_size+validation_size:]

# Normalize the pixel values
X_train = X_train / 255
X_valid = X_valid / 255
X_test = X_test / 255

# Define the model architecture
model = Sequential([
    Input(shape=(240, 640)),
    Flatten(),
    Dense(64),
    LeakyReLU(negative_slope=0.1),
    Dense(32),
    LeakyReLU(negative_slope=0.1),
    Dense(16),
    LeakyReLU(negative_slope=0.1),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size, verbose=1)

# Print model summary
model.summary()

# Get training and validation loss and accuracy
tr_loss = history.history['loss']
val_loss = history.history['val_loss']
tr_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Find the epoch with the lowest validation loss
index_loss = val_loss.index(min(val_loss))
val_lowest = val_loss[index_loss]
loss_label = f'Lowest Validation Loss: {val_lowest:.4f} at Epoch {index_loss+1}'

# Find the epoch with the highest validation accuracy
index_acc = val_acc.index(max(val_acc))
acc_highest = val_acc[index_acc]
acc_label = f'Highest Validation Accuracy: {acc_highest:.4f} at Epoch {index_acc+1}'

# Create a range of epochs for plotting
epochs_range = range(1, epochs+1)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, tr_loss, 'r', label='Training Loss')
plt.plot(epochs_range, val_loss, 'g', label='Validation Loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, tr_acc, 'r', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.png')
plt.show()

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=1)
valid_score = model.evaluate(X_valid, y_valid, verbose=1)
test_score = model.evaluate(X_test, y_test, verbose=1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Generate classification report
class_report = classification_report(y_test, y_pred_classes)
print(class_report)

# Save classification report as a figure
plt.figure(figsize=(10, 6))
plt.text(0.1, 0.5, class_report, fontsize=12, ha='left', va='center')
plt.axis('off')
plt.tight_layout()
plt.savefig('classification_report.png')
plt.show()

# Save the model architecture to JSON file
model_json = model.to_json()
with open('GestureRecognition_model.json', 'w') as json_file:
    json_file.write(model_json)
print('Model saved to disk')

# Save the model weights
model.save_weights('GestureRecognition.weights.h5')
print('Weights saved to disk')