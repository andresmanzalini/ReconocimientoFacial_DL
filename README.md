
# Reconocimiento Facial con Deep Learning


Partiendo del modelo base provisto por Keras, la red neuronal VGG16 sin la ultima capa, es posible afinarlo (fine tuning) y asi adaptarlo a nuestro problema ajustando los hiperparametros y las ultimas capas densas segun nuestro dataset. 

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

Cuando haya suficientes imagenes para identificar en un dataset balanceado, afinamos el modelo, definimos los hiperparametros segun el problema y ejecutamos los scripts vinculados al modelo:

```
python entrenarModelo.py
```

Entrena modelo y lo guarda en formato .h5 en carpeta local.

```
python usarModelo.py
```

Usa modelo guardado en disco, y predice imagenes desde el disco o desde la camara web.


![Con Ma](/imagenes/ConMa.png)


![Con el Vivi](/imagenes/Yo_V.png)


Este proyecto es la continuacion  de Deteccion de rostros con Machine learning.
La idea era darle otro enfoque al problema de reconocimiento facial. No tanto desde el aprendizaje a la fuerza bruta por parte de las maquinas, sino analizar lo que 've' desde las imagenes que capta, y como las divide en capas convolucionales y va 'aprendiendo' las caracteristicas explicitas e implicitas de las imagenes, que nosotros a simple vista no podemos. 


Hecho con
Python in Jupyter Notebook, Anaconda. 
Keras

