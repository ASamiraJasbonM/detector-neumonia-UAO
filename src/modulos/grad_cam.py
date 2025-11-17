#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para generación de mapas de calor Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualiza las regiones de la imagen que más influyen en la predicción del modelo.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

# ✅ CORREGIDO: Importaciones desde tu estructura de módulos
from .load_model import model_fun
from .preprocess_img import preprocess

def grad_cam(array, conv_layer_name="conv10_thisone"):
    """
    Genera un mapa de calor Grad-CAM para la imagen proporcionada.
    
    Args:
        array (numpy.ndarray): Imagen original como array numpy
        conv_layer_name (str): Nombre de la capa convolucional para Grad-CAM
        
    Returns:
        numpy.ndarray: Imagen con el mapa de calor superpuesto en RGB
                      o None en caso de error
    """
    try:
        print("🔥 Iniciando Grad-CAM...")
        
        # 1. PREPROCESAR la imagen
        img_preprocesada = preprocess(array)
        if img_preprocesada is None:
            print("❌ No se pudo preprocesar la imagen para Grad-CAM")
            return None
        
        # 2. CARGAR el modelo
        model = model_fun()
        if model is None:
            print("❌ No se pudo cargar el modelo para Grad-CAM")
            return None
        
        # 3. OBTENER la predicción y clase objetivo
        preds = model.predict(img_preprocesada, verbose=0)
        class_idx = np.argmax(preds[0])
        print(f"   - Clase predicha: {class_idx}")
        
        # 4. OBTENER la capa convolucional objetivo
        try:
            target_layer = model.get_layer(conv_layer_name)
        except ValueError:
            print(f"❌ No se encontró la capa '{conv_layer_name}'")
            # Intentar encontrar una capa convolucional alternativa
            target_layer = encontrar_capa_convolucional_alternativa(model)
            if target_layer is None:
                return None
        
        # 5. CALCULAR Grad-CAM usando el método original (compatible con tu código)
        output = model.output[:, class_idx]
        grads = K.gradients(output, target_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        
        # Función para obtener gradientes y salidas de la capa
        iterate = K.function([model.input], [pooled_grads, target_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([img_preprocesada])
        
        # 6. CREAR el heatmap ponderando las activaciones con los gradientes
        for i in range(conv_layer_output_value.shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # Aplicar ReLU
        
        # 7. NORMALIZAR el heatmap
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        else:
            print("⚠️  Heatmap vacío, usando valores por defecto")
            heatmap = np.ones_like(heatmap) * 0.5
        
        # 8. PREPARAR visualización
        # Redimensionar al tamaño de entrada del modelo
        heatmap = cv2.resize(heatmap, (img_preprocesada.shape[1], img_preprocesada.shape[2]))
        
        # Convertir a colores
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 9. PREPARAR imagen original
        img_original = preparar_imagen_original(array, (512, 512))
        
        # 10. SUPERPOR heatmap sobre imagen original
        alpha = 0.6  # Transparencia del heatmap
        imagen_superpuesta = cv2.addWeighted(img_original, 1-alpha, heatmap_color, alpha, 0)
        
        # 11. CONVERTIR de BGR a RGB para visualización correcta
        imagen_superpuesta_rgb = cv2.cvtColor(imagen_superpuesta, cv2.COLOR_BGR2RGB)
        
        print("✅ Grad-CAM generado exitosamente")
        return imagen_superpuesta_rgb
        
    except Exception as e:
        print(f"❌ Error crítico en Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None

def encontrar_capa_convolucional_alternativa(model):
    """
    Busca una capa convolucional alternativa si la esperada no existe.
    
    Args:
        model (tf.keras.Model): Modelo de keras
        
    Returns:
        tf.keras.layers.Layer: Capa convolucional alternativa o None
    """
    try:
        # Buscar cualquier capa convolucional
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
                print(f"🔍 Usando capa alternativa: {layer.name}")
                return layer
        
        print("❌ No se encontró ninguna capa convolucional adecuada")
        return None
    except Exception as e:
        print(f"❌ Error buscando capa alternativa: {e}")
        return None

def preparar_imagen_original(array, target_size=(512, 512)):
    """
    Prepara la imagen original para la superposición del heatmap.
    
    Args:
        array (numpy.ndarray): Imagen original
        target_size (tuple): Tamaño objetivo
        
    Returns:
        numpy.ndarray: Imagen preparada en BGR
    """
    # Redimensionar
    img = cv2.resize(array, target_size)
    
    # Convertir a BGR (formato que usa OpenCV)
    if len(img.shape) == 2:  # Escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Si ya es BGR, no hacer nada
    
    return img