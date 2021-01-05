# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ### Libraries 📚⬇

# %%
import os, cv2
import numpy as np
import pandas as pd
import random


import warnings

warnings.filterwarnings("ignore")

import albumentations as album
import tensorflow as tf

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm


DATA_DIR = "tiff"
x_train_dir = os.path.join(DATA_DIR, "train")
y_train_dir = os.path.join(DATA_DIR, "train_labels")

x_valid_dir = os.path.join(DATA_DIR, "val")
y_valid_dir = os.path.join(DATA_DIR, "val_labels")

x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "test_labels")


# %%
# class_dict = pd.read_csv("../input/massachusetts-roads-dataset/label_class_dict.csv")
class_dict = pd.read_csv("label_class_dict.csv")
# Get class names
class_names = class_dict["name"].tolist()
# Get class RGB values
class_rgb_values = class_dict[["r", "g", "b"]].values.tolist()

print("All dataset classes and their corresponding RGB values in labels:")
print("Class Names: ", class_names)
print("Class RGB values: ", class_rgb_values)

# %% [markdown]
# #### Shortlist specific classes to segment

# %%
# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ["background", "road"]

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

print("Selected classes and their corresponding RGB values in labels:")
print("Class Names: ", class_names)
print("Class RGB values: ", class_rgb_values)

# %% [markdown]
# ### Helper functions for viz. & one-hot encoding/decoding

# %%

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


# %%
class RoadsDataset(tf.keras.utils.Sequence):

    """Massachusetts Roads Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self,
        batch_size,
        images_dir,
        masks_dir,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):

        self.batch_size = batch_size

        self.image_paths = [
            os.path.join(images_dir, image_id)
            for image_id in sorted(os.listdir(images_dir))
        ]
        self.mask_paths = [
            os.path.join(masks_dir, image_id)
            for image_id in sorted(os.listdir(masks_dir))
        ]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getsingle(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __getitem__(self, i):
        indices = range(
            self.batch_size * i, min(len(self.image_paths), self.batch_size * (i + 1))
        )
        X, Y = [], []
        for j in indices:
            x, y = self.__getsingle(j)
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)


def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(
            min_height=1536, min_width=1536, always_apply=True, border_mode=0
        ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    # _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


augmented_dataset = RoadsDataset(
    32,
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(augmented_dataset) - 1)
# Get train and val dataset instances
train_dataset = RoadsDataset(
    32,
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(),
    class_rgb_values=select_class_rgb_values,
)

valid_dataset = RoadsDataset(
    1,
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(),
    class_rgb_values=select_class_rgb_values,
)

BACKBONE = "resnet50"
preprocess_input = sm.get_preprocessing(BACKBONE)
train_dataset = preprocess_input(train_dataset)

model = sm.Unet(BACKBONE, classes=2)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="epoch.{epoch:02d}.h5"
)


model.compile(
    "Adam", loss=sm.losses.categorical_crossentropy, metrics=["accuracy"]
)  # bce jaccard == IOU
model.fit(
    train_dataset,
    epochs=10,
    callbacks=[model_checkpoint_callback],
    validation_data=valid_dataset,
)

model.save('final.h5')