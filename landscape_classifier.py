import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Define label names
labelnames = ['forest', 'buildings', 'river', 'mobilehomepark', 'harbor', 'golfcourse', 'agricultural', 'runway',
                'baseballdiamond', 'overpass', 'chaparral', 'tenniscourt', 'intersection', 'airplane', 'parkinglot',
                'sparseresidential', 'mediumresidential', 'denseresidential', 'beach', 'freeway', 'storagetanks']
classes = 21
image_size = (64, 64)

# Function to load dataset
def load_dataset(root_folder):
    data = []
    labels = []

    for i, labelname in enumerate(labelnames):
        path = os.path.join(root_folder, labelname)
        print(path)
        images = os.listdir(path)

        for a in images:
            try:
                image = Image.open(os.path.join(path, a))
                image = image.resize(image_size)
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(e)

    data = np.array(data)
    labels = np.array(labels)

    # Normalize the pixel values
    data = data.astype('float32') / 255.0

    # One-hot encoding of labels
    labels = utils.to_categorical(labels)

    return data, labels

# Function to create and train the model
def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(21))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Function to preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to generate colored output image
def generate_output_image(predicted_class):
    class_colors = {
        0: [0, 0, 255],   # Water (Blue)
        1: [255, 255, 0], # Sand (Yellow)
        2: [0, 255, 0],   # Forest Vegetation (Green)
        3: [0, 255, 255], # Wetland (Cyan)
        4: [255, 0, 0],   # Roadways (Red)
        5: [255, 255, 255], # Snow/Ice (White)
        6: [128, 128, 128], # Manmade Area (Gray)
        7: [0, 128, 128], # Mobile Home Park
        8: [255, 0, 255], # Harbor
        9: [128, 0, 0],   # Golf Course
        10: [0, 128, 0],  # Agricultural
        11: [128, 128, 0],# Runway
        12: [0, 0, 128],  # Baseball Diamond
        13: [128, 0, 128],# Overpass
        14: [0, 128, 255],# Chaparral
        15: [255, 128, 0],# Tennis Court
        16: [255, 255, 128],# Intersection
        17: [128, 255, 0],# Airplane
        18: [128, 128, 255],# Parking Lot
        19: [255, 128, 128],# Sparse Residential
        20: [128, 255, 255]# Medium Residential
    }

    output_np = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            output_np[i, j] = class_colors[predicted_class]

    return Image.fromarray(output_np)


# Function to display class probabilities
def display_predictions(predicted_probs):
    for i, prob in enumerate(predicted_probs):
        print(f"{labelnames[i]}: {prob}")

def convert_to_64x64x3(arr):
    l, h, _ = arr.shape
    assert l % 64 == 0 and h % 64 == 0, "Array dimensions must be divisible by 64"

    new_shape = (l // 64, h // 64, 64, 64, 3)
    return arr.reshape(new_shape)

# Main function to handle command-line interface
def main():
    parser = argparse.ArgumentParser(description="Landcover Classification")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    # Load dataset
    root_folder = '/Users/vaidyanathaniyer/Documents/Code/landscape_classifier/UCMerced_LandUse/Images/'
    data, labels = load_dataset(root_folder)

    # Create and train the model
    input_shape = (image_size[0], image_size[1], 3)
    model = create_model(input_shape)
    model.fit(data, labels, epochs=50, batch_size=64)

    # Preprocess input image
    new_image = preprocess_image(args.image_path)

    # Predict the class of the new image
    predicted_probs = model.predict(new_image)[0]
    predicted_class = np.argmax(predicted_probs)

    print(f"Predicted class: {labelnames[predicted_class]}")
    print("Class probabilities:")
    display_predictions(predicted_probs)

if __name__ == "__main__":
    main()
