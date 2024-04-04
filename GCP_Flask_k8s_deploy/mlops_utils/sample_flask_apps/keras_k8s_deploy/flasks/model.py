from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def load_model():
    input_shape = (299, 299, 3)
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=True)

    model = Model(inputs=base_model.input, outputs=base_model.output)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics={'accuracy'})
    #model.summary()
    return model
