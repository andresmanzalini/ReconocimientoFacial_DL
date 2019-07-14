import numpy as np
import dlib
import cv2
import os
import time
import sys

# guarda fotos RGB alineadas y recortadas con DEFAULT_SIZE en la ubicacion acarada en DATASET 
# personalizar con el path al dataset y como argumento la etiqueta de la persona a identificar

DATASETS = '/Users/andresmanzalini/Documents/Datasets/'
FORMATO = '.jpg' #ver como mejorar/optimizar formato

DEFAULT_SIZE = (224,224,3) 

ETIQUETA = sys.argv[-1]

print('Etiqueta pasada como argumento ', ETIQUETA)

if not os.path.exists(DATASETS+ETIQUETA):
    print('Creo la etiqueta ', ETIQUETA)
    os.makedirs(DATASETS+ETIQUETA)
else:
    print('Ya existe la carpeta ',ETIQUETA)


# OpenCV
data_path_cv2 = cv2.__path__[0]+'/data/'
haar_type = 'haarcascade_frontalface_default.xml'

video = cv2.VideoCapture(0)
face_cascada = cv2.CascadeClassifier(data_path_cv2+haar_type)


# DLIB
path_dlibModel = os.path.dirname(os.getcwd())
predictor_model = 'shape_predictor_68_face_landmarks.dat'

land_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_dlibModel+'/'+predictor_model)


cont = 0
seguir = True

while seguir:
    ret, frame = video.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    key = cv2.waitKey(1)

    ####  OPENCV  ####   Face Cascade
    '''
    start = time.time()

    caras = face_cascada.detectMultiScale(gris, scaleFactor = 1.2, minNeighbors = 5)
    
    for (x,y,h,w) in caras:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cara = frame[y:y+h, x:x+w]
        #ver como alinear en opencv! y comparar tiempos
        end = time.time()
        print("OPENCV: ", format(end - start, '.3f'))

        #mejorar calidad de imagen a guardar en dataset.
        
        cv2.namedWindow('cara',cv2.WINDOW_FULLSCREEN)
    
        cv2.imshow('cara', cara)

        if key == ord('s'):
            nueva_foto = DATASETS+ETIQUETA+'/'+ETIQUETA+time.strftime('_%d.%m.%Y_%H:%M:%S')
            print("foto OPENCV ", nueva_foto)
            cv2.imwrite(nueva_foto+FORMATO, cara)
    '''

    ####  DLIB  ####  HOG - Histogram of Oriented Gradients
    #'''
    #start = time.time()

    caras_dlib = land_detector(gris,1)

    for cara in caras_dlib:
        x = cara.left()
        y = cara.top()
        w = cara.right() - x
        h = cara.bottom() - y
        
        landmarks = predictor(gris, cara)

        face_aligned = dlib.get_face_chip(frame, landmarks, DEFAULT_SIZE[0])
        
        #end = time.time()
        #print("DLIB aligned: ", format(end - start, '.3f'))
        
        cv2.rectangle(frame, (x,y), (cara.right(),cara.bottom()), (0,255,0), 3)
        
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(face_aligned, (x,y), 3, (255,0,0))
        
        if key == ord('s'):
            path_nueva_foto = DATASETS+ETIQUETA+'/'+ETIQUETA+time.strftime('_%d-%m-%Y_%H.%M.%S')
            nombre_foto = ETIQUETA+time.strftime('_%d-%m-%Y_%H.%M.%S')
            print("Sacaste nueva foto dlib ", nombre_foto)
            cv2.imwrite(path_nueva_foto+FORMATO, face_aligned)
            cont += 1

        cv2.imshow('Cara alineada', face_aligned)
        #'''

    #cv2.imshow('Frame',frame)
    
    if key == ord ('q'):
        print('------------------------------------------')
        print("{}, sacaste {} fotos".format(ETIQUETA,cont))
        seguir=False
        

video.release()
cv2.destroyAllWindows()

