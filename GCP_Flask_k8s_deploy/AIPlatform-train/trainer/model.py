from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
def sonar_model():
    base_model = VGG16(weights='imagenet', include_top=True, input_tensor=Input(shape=(224,224,3)))
    x = base_model.get_layer(index=-5).output
    x = Dropout(rate=0.3)(x)
    x = GlobalAveragePooling2D()(x)
    o = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=o)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    return model
