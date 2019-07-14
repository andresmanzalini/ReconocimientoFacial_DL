from keras.layers import Input,Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Lambda

from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dlib

from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model


default_size=(224,224,3)
batch_size=16
nombre_modelo='reconocimientoFacial.h5'


generador = ImageDataGenerator(validation_split=0.25,
                               rescale=1./255,
                               shear_range = 0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)


training_set = generador.flow_from_directory('/Users/andresmanzalini/Documents/Datasets',
                                             target_size = (224,224),
                                             batch_size=16,
                                             class_mode='categorical') 

validation_set = generador.flow_from_directory('/Users/andresmanzalini/Documents/Datasets',
                                                   target_size = (224,224),
                                                   batch_size=16,
                                                   class_mode='categorical')


cant_ids = training_set.num_classes
etiquetas = training_set.class_indices
tags = list(etiquetas.keys())



def get_modelo():
    vgg = VGG16(input_shape=(default_size), weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    c1 = Conv2D(256, (3,3), activation='relu')(vgg.output)
    #c2 = Conv2D(256, (3,3), activation='relu')(c1)
    p1 = MaxPooling2D(2,2)(c1)

    f = Flatten()(p1) 

    d1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(f)
    
    dr = Dropout(0.25)(d1)

    d2 = Dense(256, activation='relu')(dr)

    out = Dense(cant_ids, activation='softmax')(d2)

    model = Model(inputs=vgg.input, outputs=out) 
    
    return model


def compilar_entrenar(modelo):
    modelo.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    modelo.summary()

    h = modelo.fit_generator(generator=training_set,
                             validation_data=validation_set,
                             epochs=40,
                             steps_per_epoch=len(training_set),
                             validation_steps=len(validation_set))

    return modelo, h



def guardar_modelo(modelo):
    modelo.save(nombre_modelo)  # HDF5 file
    del modelo


def dibujar_historia(h):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



if __name__ == "__main__":
    modelo = get_modelo()
    modelo, h = compilar_entrenar(modelo)
    eval = modelo.evaluate_generator(generator=validation_set, steps=len(validation_set))
    print('Evaluacion de modelo: ', eval)
    dibujar_historia(h)
    guardar_modelo(modelo)
    