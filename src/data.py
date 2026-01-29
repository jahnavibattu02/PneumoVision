from tensorflow.keras.preprocessing.image import ImageDataGenerator

def make_generators(data_dir: str, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
        horizontal_flip=False,  # chest X-rays usually avoid flip
        brightness_range=(0.9, 1.1),
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        directory=f"{data_dir}/train",
        target_size=img_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        directory=f"{data_dir}/val",
        target_size=img_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
    )

    test_gen = val_datagen.flow_from_directory(
        directory=f"{data_dir}/test",
        target_size=img_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
