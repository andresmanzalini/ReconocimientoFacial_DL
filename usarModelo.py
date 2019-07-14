import dlib
import cv2
import os
import time
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K


PATH_DATASET = '/Users/andresmanzalini/Documents/Datasets/'


def cargar_modelo():
    model = load_model('reconocimientoFacial.h5')
    return model


def get_etiquetas():
    generador = ImageDataGenerator()
    gen = generador.flow_from_directory(PATH_DATASET)                                               
    etiquetas = gen.class_indices
    tags = list(etiquetas.keys()) 
    print(tags)
    return tags


def prediccion_disco(modelo):
    im_yo = '/Users/andresmanzalini/Documents/Andres_Prueba.jpg'
    im_v = '/Users/andresmanzalini/Documents/V.jpg'

    img = image.load_img(im_yo, target_size=(224, 224))
    print(img)
    x = image.img_to_array(img)
    print(x.dtype)
    x = np.expand_dims(x, axis=0)

    prediccion = model.predict(x)
    pred = np.argmax(prediccion) 

    preds = prediccion[0,:]
    #print('predicciones ',preds)
    print("prediccion: {0:.3f}". format(preds[pred]))




def prediccion_webcam(modelo, tags):
    #OpenCV
    data_path_cv2 = cv2.__path__[0]+'/data/'
    #haar_type = 'haarcascade_frontalface_default.xml'

    video = cv2.VideoCapture(0)
    #face_cascada = cv2.CascadeClassifier(data_path_cv2+haar_type)

    
    #DLIB
    path_DLIBmodel = os.path.dirname(os.getcwd())+'/shape_predictor_68_face_landmarks.dat'
    land_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_DLIBmodel)

    
    seguir = True
    while seguir:
        ret, frame = video.read()
        #gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #sobreescribo formato de salida de GBR a RGB 
        gris = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2GRAY)

        key = cv2.waitKey(1) 

        caras_dlib = land_detector(gris,1)

        for cara in caras_dlib:
            x = cara.left()
            y = cara.top()
            w = cara.right() - x
            h = cara.bottom() - y

            landmarks = predictor(gris, cara)
            face_aligned = dlib.get_face_chip(frame, landmarks, 224)

            face_aligned_32 = np.asarray(face_aligned, dtype='float32') 
            #print('face aligned 32 ', face_aligned_32)
            im_expand = np.expand_dims(face_aligned_32, axis=0) 
            #print('expand ', im_expand.shape)
            im_normalizada = (im_expand - np.min(im_expand)) / (np.max(im_expand) - np.min(im_expand))
            #normalizando predice probabilidades!
            
            prediccion = model.predict(im_normalizada)

            pred = np.argmax(prediccion) 
            proba = prediccion[:,pred]
            tag = tags[pred]
            
            #print('PREDICCION ', prediccion)
            #print('pred ',pred)
            #print('prob ',proba)
            
            if proba > .7: # & reconoce al mismo id durante 2s
                print('{}: {}'.format(tag, proba))
                cv2.putText(frame, tag, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.rectangle(frame, (x,y), (cara.right(),cara.bottom()), (0,255,0), 3)

            for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x,y), 3, (255,0,0))

            #cv2.imshow('Cara alineada', face_aligned)

        cv2.imshow('Frame',frame)

        if key == ord ('q'):
            seguir=False


    video.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    model = cargar_modelo()
    tags = get_etiquetas() 
    prediccion_webcam(model, tags)

