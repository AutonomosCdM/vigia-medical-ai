"""
Utilidades para procesamiento de imágenes en LPP-Detect.

Este módulo proporciona funciones auxiliares para manipulación, 
listado y visualización de imágenes para el sistema LPP-Detect.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Configuración de logging
logger = logging.getLogger('lpp-detect.image_utils')

# Extensiones válidas para imágenes
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

def list_image_files(directory):
    """
    Lista todas las imágenes en un directorio.
    
    Args:
        directory: Ruta al directorio
        
    Returns:
        list: Lista de objetos Path para cada imagen
    """
    if not os.path.isdir(directory):
        logger.error(f"El directorio {directory} no existe")
        return []
    
    directory_path = Path(directory)
    image_files = []
    
    for ext in VALID_IMAGE_EXTENSIONS:
        image_files.extend(directory_path.glob(f'*{ext}'))
        image_files.extend(directory_path.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)

def save_detection_result(image, detection_results, output_path):
    """
    Guarda una imagen con las anotaciones de detección.
    
    Args:
        image: Imagen como array NumPy
        detection_results: Resultados de detección
        output_path: Ruta para guardar la imagen resultante
        
    Returns:
        str: Ruta a la imagen guardada
    """
    # Asegurarse de que la imagen esté en formato correcto
    if image.dtype == np.float32 and np.max(image) <= 1.0:
        # Convertir de vuelta a uint8 para dibujo
        display_image = (image * 255).astype(np.uint8)
    else:
        display_image = image.copy()
    
    # Dibujar cada detección
    for detection in detection_results["detections"]:
        bbox = detection["bbox"]
        stage = detection["stage"]
        confidence = detection["confidence"]
        
        # Coordenadas como enteros
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color según etapa (verde=1, amarillo=2, naranja=3, rojo=4)
        colors = [
            (0, 255, 0),    # Verde para Etapa 1
            (0, 255, 255),  # Amarillo para Etapa 2
            (0, 165, 255),  # Naranja para Etapa 3
            (0, 0, 255)     # Rojo para Etapa 4
        ]
        
        # Seleccionar color según etapa (con manejo de índice fuera de rango)
        color_idx = min(stage, len(colors) - 1)
        color = colors[color_idx]
        
        # Dibujar cuadro delimitador
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
        
        # Preparar texto
        text = f"LPP Etapa {stage}: {confidence:.2f}"
        
        # Fondo para texto
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(
            display_image, 
            (x1, y1 - text_size[1] - 10), 
            (x1 + text_size[0], y1), 
            color, 
            -1
        )
        
        # Texto
        cv2.putText(
            display_image, 
            text, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 0), 
            1, 
            cv2.LINE_AA
        )
    
    # Añadir texto de timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        display_image, 
        f"LPP-Detect {timestamp}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 0), 
        2, 
        cv2.LINE_AA
    )
    
    # Asegurar que el directorio existe
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar imagen
    cv2.imwrite(str(output_path), display_image)
    logger.info(f"Imagen con anotaciones guardada en {output_path}")
    
    return str(output_path)

def save_detection_visualization(image, detection_results, output_path):
    """Alias for save_detection_result for compatibility."""
    return save_detection_result(image, detection_results, output_path)

def anonymize_image(image):
    """Anonymize an image by blurring sensitive areas."""
    # Simple implementation - return image as-is for now
    return image

def crop_lpp_region(image, bbox, padding=20):
    """
    Recorta la región de LPP de una imagen con padding.
    
    Args:
        image: Imagen como array NumPy
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding alrededor del bbox
        
    Returns:
        numpy.ndarray: Imagen recortada
    """
    x1, y1, x2, y2 = map(int, bbox)
    height, width = image.shape[:2]
    
    # Aplicar padding con límites de imagen
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    
    # Recortar imagen
    cropped = image[y1:y2, x1:x2]
    
    return cropped

def is_valid_image(image_path):
    """
    Valida si un archivo es una imagen válida.
    
    Args:
        image_path: Ruta al archivo de imagen
        
    Returns:
        dict: Resultado de validación con campos 'valid', 'format', etc.
    """
    if not image_path or not os.path.exists(image_path):
        return {'valid': False, 'error': 'File not found'}
    
    try:
        # Verificar extensión
        ext = Path(image_path).suffix.lower()
        if ext not in VALID_IMAGE_EXTENSIONS:
            return {'valid': False, 'error': 'Invalid extension'}
        
        # Intentar leer imagen
        image = cv2.imread(str(image_path))
        if image is None:
            return {'valid': False, 'error': 'Cannot read image'}
        
        return {
            'valid': True,
            'format': ext[1:].upper(),
            'size': (image.shape[1], image.shape[0]),
            'channels': image.shape[2] if len(image.shape) > 2 else 1
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def generate_grid_view(images, labels=None, cols=4, cell_size=(200, 200)):
    """
    Genera una vista en cuadrícula de múltiples imágenes.
    
    Args:
        images: Lista de imágenes
        labels: Lista de etiquetas para cada imagen
        cols: Número de columnas en la cuadrícula
        cell_size: Tamaño de cada celda (ancho, alto)
        
    Returns:
        numpy.ndarray: Imagen de cuadrícula
    """
    # Determinar filas y columnas
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Crear lienzo
    grid = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)
    
    # Rellenar con imágenes
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        # Redimensionar imagen
        if img.shape[0] != cell_size[1] or img.shape[1] != cell_size[0]:
            img_resized = cv2.resize(img, cell_size)
        else:
            img_resized = img
            
        # Calcular posición
        row = i // cols
        col = i % cols
        
        # Colocar imagen
        grid[row*cell_size[1]:(row+1)*cell_size[1], 
             col*cell_size[0]:(col+1)*cell_size[0]] = img_resized
        
        # Añadir etiqueta si existe
        if labels and i < len(labels):
            cv2.putText(
                grid, 
                str(labels[i]), 
                (col*cell_size[0] + 5, row*cell_size[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
    
    return grid
