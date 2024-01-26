from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.image import resize
import numpy as np

def intensityModel():
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in vgg_model.layers:
        layer.trainable = False
    
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    
    return model

def estimateIntensity(model, img):
    model.load_weights('./models/' + 'vgg16.h5')
    img = resize(img, size = (224, 224))
    img = np.expand_dims(img, axis = 0)
    pred = model.predict(img/255)[0][0]*100

    pred = pred * 1.852

    return round(pred, 2)