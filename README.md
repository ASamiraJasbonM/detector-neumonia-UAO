# 🩺 Sistema de Detección de Neumonía con Deep Learning

Sistema de inteligencia artificial para el apoyo al diagnóstico de neumonía en imágenes radiográficas utilizando redes neuronales convolucionales y Grad-CAM para explicabilidad.

## ✨ Características

- **Clasificación Automática**: Detecta neumonía bacteriana, viral y casos normales
- **Interfaz Gráfica Intuitiva**: Fácil de usar para profesionales médicos
- **Grad-CAM Integration**: Mapas de calor que muestran áreas relevantes para el diagnóstico
- **Reportes en PDF**: Generación automática de reportes médicos
- **Sistema Modular**: Arquitectura limpia y mantenible

## 🚀 Instalación

### Requisitos
- Python 3.8+
- TensorFlow 2.11+
- OpenCV 4.7+

### Paso 1: Instalación rápida
```bash
git clone [https://github.com/ASamiraJasbonM/detector-neumonia-UAO.git]
cd detector-neumonia
pip install -r requirements.txt
```

### Paso 2: Desccargar y mover modelo de clasificación 
En caso de no poder descarcar modelo, no se podrá realizar predicciones.
- conv_MLP_84.h5
- Mover archivo conv_MLP_84.h5 a C:\Users\[usuario]\detector-neumonia-UAO\models\


### Paso 3: Construir y ejecutar la imagen en docker
```bash
docker build -t detector-neumonia .
docker run -p 5000:5000 detector-neumonia
```

### Ejecución: 
```bash
python main.py

```

### Prueba de funcionamiento:
- Es necesario probar el funcionamiento de los componentes para asegurar que ha sido exitosa la instalación, aunmque este paso se puede saltar si se ejecuta correctamente.
```bash
python test_integrator.py
python test_quick.py
python test_simple.py
```
