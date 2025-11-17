#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para carga y gestión del modelo de red neuronal convolucional
Modelo principal: 'conv_MLP_84.h5'
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np

# ✅ CORREGIDO: Configuración de TensorFlow en un solo lugar
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

def model_fun():
    """
    Función principal para cargar el modelo pre-entrenado.
    Busca el modelo en varias ubicaciones posibles.
    
    Returns:
        tf.keras.Model: Modelo cargado listo para predicción o None en caso de error
    """
    try:
        # ✅ CORREGIDO: Rutas relativas a tu estructura de proyecto
        possible_paths = [
            'models/conv_MLP_84.h5',           # Desde raíz
            '../models/conv_MLP_84.h5',        # Desde src/modulos
            '../../models/conv_MLP_84.h5',     # Desde otras ubicaciones
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("No se encontró el modelo en ninguna ubicación posible")
        
        print(f"🔄 Cargando modelo desde: {model_path}")
        
        # Cargar modelo sin compilar inicialmente
        model = load_model(model_path, compile=False)
        
        # ✅ MEJORADO: Compilar con configuración optimizada
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Validar que el modelo esté listo
        if validar_modelo_cargado(model):
            print(f"✅ Modelo cargado exitosamente: {model_path}")
            print(f"   - Capas: {len(model.layers)}")
            print(f"   - Parámetros: {model.count_params():,}")
            return model
        else:
            print("❌ El modelo cargado no pasó la validación")
            return None
        
    except Exception as e:
        print(f"❌ Error crítico cargando el modelo: {e}")
        return None

def validar_modelo_cargado(model):
    """
    Valida que el modelo cargado tenga la estructura esperada.
    
    Args:
        model (tf.keras.Model): Modelo a validar
        
    Returns:
        bool: True si el modelo es válido, False en caso contrario
    """
    if model is None:
        return False
    
    try:
        # Verificar que tenga la capa necesaria para Grad-CAM
        layer_names = [layer.name for layer in model.layers]
        
        if 'conv10_thisone' not in layer_names:
            print("⚠️  No se encontró la capa 'conv10_thisone' para Grad-CAM")
            print(f"   Capas disponibles: {[name for name in layer_names if 'conv' in name]}")
            # No retornar False, solo advertir
        
        # Probar una predicción de prueba
        test_input = np.random.rand(1, 512, 512, 1).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        
        if test_output.shape[1] == 3:  # Debe tener 3 clases
            print("✅ Modelo validado: arquitectura correcta")
            return True
        else:
            print(f"❌ Modelo tiene {test_output.shape[1]} clases, se esperaban 3")
            return False
            
    except Exception as e:
        print(f"❌ Error validando modelo: {e}")
        return False

# ✅ MANTENIDO: Funciones adicionales para futuras extensiones
def load_custom_model(model_path):
    """Carga un modelo desde una ruta específica"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Archivo no encontrado: {model_path}")
        
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print(f"✅ Modelo personalizado cargado: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Error cargando modelo personalizado: {e}")
        return None