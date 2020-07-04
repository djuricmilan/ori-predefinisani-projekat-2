from utility import *
from hyperparameters import  *
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Softmax, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

def read_labels(path):
    df = pd.read_csv(path)
    dataframe = df.query("Label == 'Normal' | Label_1_Virus_category == 'bacteria' | Label_1_Virus_category == 'Virus'")
    excluded_count = df.shape[0] - dataframe.shape[0]

    dataframe["_label"] = "Normal"
    dataframe.loc[dataframe['Label_1_Virus_category'] == 'bacteria', '_label'] = "Bacteria"
    dataframe.loc[dataframe['Label_1_Virus_category'] == 'Virus', '_label'] = "Virus"

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


def split_train_test(dataframe):
    if not os.path.exists(TEST_DIR_NAME):
        os.makedirs(TEST_DIR_NAME)

    #copy all images back to the main folder
    for file in os.listdir(TEST_DIR_NAME):
        src = os.path.join(TEST_DIR_NAME, file)
        shutil.move(src, DIR_NAME)

    dataframe = dataframe.sample(frac=1).reset_index(drop=True) #shuffle dataframe first
    msk = np.random.rand(len(dataframe)) < 0.8 #20% of data is for testing
    train = dataframe[msk]
    test = dataframe[~msk]

    # copy new test data to the test folder
    for index, row in test.iterrows():
        src = os.path.join(DIR_NAME, row["X_ray_image_name"])
        shutil.move(src, TEST_DIR_NAME)
    print("\n*****TRAIN AND TEST DATASET CREATED*****")
    return train, test

def main():
    #read dataframe with labels
    print("******TRAINING DATA OVERVIEW******")
    dataframe = read_labels(LABELS_PATH)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)  # shuffle dataframe first

    #split train and test
    print("******TEST DATA OVERVIEW******")
    test = read_labels(TEST_LABELS_PATH)
    #dataframe, test = split_train_test(dataframe)

    # prepare training and testing augmentation configuration
    data_generator = ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=0.2
    )
    test_generator = ImageDataGenerator(
        rescale=1. / 255.
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
        color_mode='grayscale'
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
        color_mode='grayscale'
    )
    test_iterator = test_generator.flow_from_dataframe(
        dataframe=test,
        directory=TEST_DIR_NAME,
        x_col="X_ray_image_name",
        y_col="_label",
        class_mode="categorical",
        batch_size=1,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode='grayscale'
    )

    #compute class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight('balanced', np.unique(dataframe['_label']), dataframe['_label'])
    class_weights = dict(enumerate(class_weights))
    print(class_weights)

    #construct model
    model = Sequential()

    #feature extractors - convolutional and pooling layers
    model.add(Conv2D(filters=32, kernel_size=(3,3), use_bias=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    #classifier
    model.add(Flatten())

    model.add(Dense(64))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))

    model.add(Dense(units=3))
    model.add(Softmax())

    #set optimizer and losss function
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    print(model.summary())

    #training
    mcp_save = ModelCheckpoint('weights/best_val_loss_model2.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_lr=0.00001)

    steps_train = train_iterator.n // train_iterator.batch_size
    steps_valid = validate_iterator.n // validate_iterator.batch_size
    history = model.fit_generator(
        generator=train_iterator,
        steps_per_epoch=steps_train,
        validation_data=validate_iterator,
        validation_steps=steps_valid,
        epochs=30,
        callbacks=[mcp_save, reduce_lr_loss],
        class_weight=class_weights
    )

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    steps_test = test_iterator.n // test_iterator.batch_size
    test_iterator.reset()

    #load best model
    model.load_weights('weights/best_val_loss_model2.h5')

    #evaluation
    results = model.evaluate_generator(
        generator=test_iterator,
        steps=steps_test,
    )
    print("Test accuraccy:", str(results[1]))

if __name__ == '__main__':
    main()