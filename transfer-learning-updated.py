#Download the train and test dataset from this site - https://zenodo.org/records/5226945 Then arrange the datasets as C:\Users\Janhavi\Downloads\cats_dogs_light\train\cat ,C:\Users\Janhavi\Downloads\cats_dogs_light\train\dog,C:\Users\Janhavi\Downloads\cats_dogs_light\test\cat,C:\Users\Janhavi\Downloads\cats_dogs_light\test\dog
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define directories
train_dir = r'C:\Users\Janhavi\Downloads\cats_dogs_light\train'
validation_dir = r'C:\Users\Janhavi\Downloads\cats_dogs_light\test'

# Check if the directories contain subdirectories for classes
print(f"Train directory exists: {os.path.isdir(train_dir)}")
print(f"Validation directory exists: {os.path.isdir(validation_dir)}")
print(f"Found {len(os.listdir(train_dir))} subdirectories in the train directory.")
print(f"Found {len(os.listdir(validation_dir))} subdirectories in the validation directory.")

# Helper function to check for corrupted or non-image files
def check_images(directory):
    print(f"Checking files in {directory}...")
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify the image is not corrupted
                except (IOError, SyntaxError) as e:
                    print(f"Removing non-image or corrupted file: {file_path}")
                    os.remove(file_path)  # Remove corrupted or invalid files

# Check and clean both train and validation directories
check_images(train_dir)
check_images(validation_dir)

# Load pre-trained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # Exclude the classification head
    weights='imagenet'
)

# Unfreeze some layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Unfreeze all but the last 10 layers
    layer.trainable = False

# Add custom classification head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),  # Reduced neurons for simplicity
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Use lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Create ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Horizontal flip
    fill_mode='nearest'  # Fill missing pixels
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Binary classification (cats vs dogs)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Binary classification (cats vs dogs)
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=[early_stopping, lr_scheduler]
)

# Visualize training progress
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Epochs')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epochs')

plt.show()
