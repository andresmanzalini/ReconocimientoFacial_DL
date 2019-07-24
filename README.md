
# Reconocimiento Facial con Deep Learning


Partimos del modelo base provisto por Keras, la red neuronal VGG16, sin la ultima capa.
Es posible afinarlo (fine tuning) para adaptarlo a nuestro problema, ajustando los hiperparametros y las ultimas capas densas segun nuestro dataset, dejando el modelo listo para compilar y entrenar. 

Al finalizar el entrenamiento, muestra las metricas y guarda el modelo en formato .h5 en la carpeta local. 
Podemos usarlo con la camara web.


Para personalizar el problema el script detectarProcesar_cara.py se encarga de la deteccion, el preprocesamiento y la carga al dataset de las imagenes de la cara asociada a la identificacion.

Con el dataset balanceado, podemos entrenar y lograr resultados aceptables.

Cuando termina de entrenar muestra los resultados por pantalla. Al cerrarlos se guarda el modelo.

Por ultimo, predecir resultados a partir de la camara web. 

**Aclaracion**: al principio con pocas imagenes de prueba en el dataset, el algoritmo no aprende bien.
A medida que mejora el dataset, mejor aprende el algoritmo.



### Prerrequisitos

Instalar requerimientos.txt y descargar los pesos del archivo dlib entrenado shape_predictor_68_face_landmarks.dat
en https://github.com/davisking/dlib-models



### Instalacion de entorno virtual
```py
conda create -n envReconocimientoFacial python
```
```py
pip install -r requerimientos.txt
```


### Ejemplo
```py
python detectarProcesar_cara.py Andres
```
```
python detectarProcesar_cara.py Martin 
```
...

lo hago algunas veces mas por id

...

Abro y ejecuto el jupyter notebook .ipynb para entender los pasos seguidos.

Cuando tengo suficientes imagenes para identificar y un dataset balanceado, afino el modelo, defino los hiperparametros segun nuestro problema y ejecuto los scripts vinculados al modelo:

```
python entrenarModelo.py
```

Entrena modelo y lo guarda en formato .h5 en carpeta local.

```
python usarModelo.py
```

Usa modelo guardado en disco, y predice imagenes desde el disco o desde la camara web.

ver como subir fotos de muestra
![Imagenes de prueba](/imagenes/Yo_V.jpg)


Este proyecto es la continuacion  de Deteccion de rostros con Machine learning.
La idea era darle otro enfoque al problema de reconocimiento facial. No tanto desde el aprendizaje de los patrones por parte de la maquina, sino analizar lo que 've' desde las imagenes que capta, y como las divide en capas convolucionales y va 'aprendiendo' las caracteristicas explicitas e implicitas de las imagenes, que nosotros a simple vista no podemos. 


Hecho con
Python in Jupyter Notebook, Anaconda. 
Keras


Licencia
creditos a Keras
licencia de dlib


La inspiracion surgio al darme cuenta que las computadoras y celulares tienen demasiado acceso a nosotros y a nuestra vida personal. Por lo tanto, obtienen una perspectiva mas completa y exacta de nostros mismos de la que ni nos podemos imaginar. 
Por suerte los dispostivos personales aun no soportan la carga computacional para funcionar asi en tiempo real, pero los algoritmos que controlan nuestro historial si.
