import numpy as np
import pandas as pd
from models import VGG16Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Data loading
X_train = pd.read_pickle('Datasets/train.pkl')[..., np.newaxis]
y_train = pd.read_csv('Datasets/train_y.csv', index_col=0)
X_test = pd.read_pickle('Datasets/test.pkl')[..., np.newaxis]

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2)
datagen.fit(X_train)
datagen.fit(X_test)

# One-hot encoding
y_train = to_categorical(y_train)

# Training and validation split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Model training
model = VGG16Model()
callback = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1)
model.fit(datagen.flow(X_train, y_train, batch_size=64), validation_data=(X_valid, y_valid), epochs=160, callbacks=[callback], verbose=2)

# Model testing
predictions = pd.DataFrame(np.argmax(model.predict(X_test), axis=1), columns=['label'])
predictions.index.name = 'id'
predictions.to_csv('results.csv')