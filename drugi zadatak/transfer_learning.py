import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers  import *
from hyperparameters import *
from utility import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def read_labels():
    df = pd.read_csv(LABELS_PATH)
    dataframe = df.query("Label == 'Normal' | Label_1_Virus_category == 'bacteria' | Label_1_Virus_category == 'Virus'")
    excluded_count = df.shape[0] - dataframe.shape[0]

    dataframe["_label"] = "Normal"
    dataframe.loc[dataframe['Label_1_Virus_category'] == 'bacteria', '_label'] = "Bacteria"
    dataframe.loc[dataframe['Label_1_Virus_category'] == 'Virus', '_label'] = "Virus"
    #dataframe["_label"] = pd.get_dummies(dataframe["_label"]).values.tolist()


    normal_count = dataframe.loc[dataframe["Label"] == "Normal"].shape[0]
    bacteria_count = dataframe.loc[dataframe["Label_1_Virus_category"] == "bacteria"].shape[0]
    virus_count = dataframe.loc[dataframe["Label_1_Virus_category"] == "Virus"].shape[0]

    print("\n*****NUMBER OF SAMPLES READ FROM CSV*****")
    print("Normal count:", str(normal_count))
    print("Bacteria count:", str(bacteria_count))
    print("Virus count:", str(virus_count))
    print("Excluded: ", str(excluded_count))
    print("Total number of training samples: ", str(dataframe.shape[0]))
    return dataframe

def main():
    #read dataframe with labels
    dataframe = read_labels()
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)  # shuffle dataframe first


    # prepare training and testing augmentation configuration
    data_generator = ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        rotation_range=40,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    test_generator = ImageDataGenerator(
        rescale=1. / 255.,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )

    #prepare train, validate and test iterators
    train_iterator = data_generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=DIR_NAME,
        x_col="X_ray_image_name",
        y_col="_label",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        subset="training",
    )
    validate_iterator = data_generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=DIR_NAME,
        x_col="X_ray_image_name",
        y_col="_label",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        subset="validation",
    )

    base_model = ResNet50(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), weights='imagenet', include_top=False)

    # freeze all layers in the base model
    base_model.trainable = False

    # un-freeze the BatchNorm layers
    for layer in base_model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True

    inputs = base_model.input
    outputs = base_model.output
    outputs = GlobalAveragePooling2D()(outputs)
    outputs = Dense(64, activation='relu')(outputs)
    outputs = Dense(3, activation='softmax')(outputs)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['acc'])
    print(model.summary())

    # training
    steps_train = train_iterator.n // train_iterator.batch_size
    steps_valid = validate_iterator.n // validate_iterator.batch_size
    model.fit_generator(
        generator=train_iterator,
        steps_per_epoch=steps_train,
        validation_data=validate_iterator,
        validation_steps=steps_valid,
        epochs=200
    )

if __name__ == '__main__':
    main()