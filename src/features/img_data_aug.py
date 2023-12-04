from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Directory containing your original images
original_data_dir = 'D:\\github\\ml-img-gen\\data\\pokemon_jpg'

# Directory where augmented images will be saved
augmented_data_dir = 'D:\\github\\ml-img-gen\\data\\pokemon_aug_small'

# Create the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the original images and generate augmented images
for root, dirs, files in os.walk(original_data_dir):
    for file in files:
        img_path = os.path.join(root, file)
        img = image.load_img(img_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_data_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i > 100:  # Generate a few augmented images per original image
                break

# Visualize some augmented images
# sample_augmented_img_path = os.path.join(augmented_data_dir, 'aug_0_0.jpeg')
# sample_augmented_img = image.load_img(sample_augmented_img_path)
#
# plt.imshow(sample_augmented_img)
# plt.show()