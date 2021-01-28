from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

def VGG16Model():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(64, 64, 1)),
        Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.2),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')])
    
    model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.8, decay=0.00003125), loss=categorical_crossentropy, metrics=['accuracy'])
    
    return model