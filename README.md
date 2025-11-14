Detector de Neumonía – Proyecto UAO

Este proyecto consiste en una herramienta gráfica desarrollada en Python que permite cargar una radiografía de tórax (DICOM o JPG), mostrarla en pantalla y ejecutar un modelo de deep learning para predecir neumonía (bacteriana, viral o normal).
El sistema utiliza Grad-CAM para generar un mapa de calor que indica las zonas relevantes analizadas por la red.

Funcionalidades Principales

Cargar imágenes de radiografía en formatos .dcm, .jpeg, .jpg, .png
Previsualización de la imagen cargada
Predicción usando un modelo CNN entrenado previamente
Visualización de mapa de calor (Grad-CAM)
Exportación de resultados a CSV
Exportación de reporte en PDF
Interfaz hecha con Tkinter

Mejoras realizadas por el estudiante

Durante la depuración del repositorio se encontraron y solucionaron los siguientes errores:

Incompatibilidad de versión de Python
TensorFlow no funciona en Python 3.12+.
Se configuró Python 3.11, versión compatible.
Problemas al crear el entorno virtual
El sistema impedía crear el venv correctamente.
Se creó y activó correctamente para garantizar dependencias aisladas.
Faltaban librerías esenciales
El código intentaba usar PIL y dicom sin importarlas.
Se corrigió importando las respectivas líbrerias
Lectura incorrecta de imágenes
La función load_img_file siempre leía como DICOM.
Se añadió manejo adecuado para JPG/PNG.

Trabajo realizado como parte del proyecto académico de la Universidad
Autónoma de Occidente (UAO).
Estudiantes:
1. César Nieto
2.
3.
