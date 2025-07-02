"""
Tests para las utilidades de procesamiento de imágenes.

Verifica que las funciones auxiliares para manipulación, listado
y visualización de imágenes funcionen correctamente.
"""

import os
import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path

from utils.image_utils import (list_image_files, save_detection_result, 
                              crop_lpp_region, generate_grid_view)

# Tests
def test_list_image_files():
    """Verifica que la función liste correctamente las imágenes de un directorio."""
    # Crear directorio temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        # Crear imágenes temporales de diferentes tipos
        img1_path = os.path.join(temp_dir, "test1.jpg")
        img2_path = os.path.join(temp_dir, "test2.PNG")  # Mayúsculas
        img3_path = os.path.join(temp_dir, "test3.jpeg")
        txt_path = os.path.join(temp_dir, "test4.txt")  # No imagen
        
        # Crear archivos dummy
        for path in [img1_path, img2_path, img3_path, txt_path]:
            with open(path, 'w') as f:
                f.write("dummy")
        
        # Listar imágenes
        image_files = list_image_files(temp_dir)
        
        # Verificar resultados
        assert len(image_files) == 3  # 3 imágenes, excluyendo .txt
        assert any(str(img_path).endswith("test1.jpg") for img_path in image_files)
        assert any(str(img_path).endswith("test2.PNG") for img_path in image_files)
        assert any(str(img_path).endswith("test3.jpeg") for img_path in image_files)
        assert not any(str(img_path).endswith("test4.txt") for img_path in image_files)

def test_list_image_files_empty_dir():
    """Verifica el comportamiento con un directorio vacío."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Listar imágenes en directorio vacío
        image_files = list_image_files(temp_dir)
        
        # Verificar que la lista está vacía
        assert len(image_files) == 0

def test_list_image_files_nonexistent_dir():
    """Verifica el comportamiento con un directorio inexistente."""
    # Directorio que no existe
    non_existent_dir = "/path/to/nonexistent/directory"
    
    # Listar imágenes
    image_files = list_image_files(non_existent_dir)
    
    # Verificar que la lista está vacía
    assert len(image_files) == 0

def test_save_detection_result():
    """Verifica que se guarde correctamente una imagen con anotaciones."""
    # Crear imagen de prueba
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = [200, 200, 200]  # Fondo gris
    
    # Resultados de detección
    detection_results = {
        "detections": [
            {
                "bbox": [50, 50, 150, 100],
                "confidence": 0.8,
                "stage": 1,
                "class_name": "LPP-Stage1"
            },
            {
                "bbox": [200, 150, 250, 200],
                "confidence": 0.7,
                "stage": 3,
                "class_name": "LPP-Stage3"
            }
        ]
    }
    
    # Crear directorio temporal para guardar resultado
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_output.jpg")
        
        # Guardar imagen con anotaciones
        result_path = save_detection_result(img, detection_results, output_path)
        
        # Verificar que el archivo existe
        assert os.path.exists(result_path)
        
        # Cargar la imagen guardada
        saved_img = cv2.imread(result_path)
        
        # Verificar que la imagen no está vacía
        assert saved_img is not None
        assert saved_img.shape == (300, 300, 3)
        
        # En un test real, podríamos verificar que los rectángulos se dibujaron
        # comparando píxeles específicos, pero es complejo y frágil

def test_crop_lpp_region():
    """Verifica que se recorte correctamente una región de LPP."""
    # Crear imagen de prueba
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = [200, 200, 200]  # Fondo gris
    
    # Dibujar un rectángulo rojo en la región a recortar
    x1, y1, x2, y2 = 100, 100, 200, 150
    img[y1:y2, x1:x2] = [0, 0, 255]  # Rectángulo rojo
    
    # Recortar con padding 0
    cropped = crop_lpp_region(img, [x1, y1, x2, y2], padding=0)
    
    # Verificar dimensiones del recorte
    assert cropped.shape == (50, 100, 3)  # alto=150-100=50, ancho=200-100=100
    
    # Verificar que todos los píxeles son rojos
    assert np.all(cropped[:, :, 0] == 0)  # B=0
    assert np.all(cropped[:, :, 1] == 0)  # G=0
    assert np.all(cropped[:, :, 2] == 255)  # R=255
    
    # Recortar con padding 10
    cropped_padded = crop_lpp_region(img, [x1, y1, x2, y2], padding=10)
    
    # Verificar dimensiones con padding
    assert cropped_padded.shape == (70, 120, 3)  # alto=50+20=70, ancho=100+20=120

def test_generate_grid_view():
    """Verifica la generación de una vista en cuadrícula de imágenes."""
    # Crear imágenes de prueba de diferentes colores
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[:, :] = [255, 0, 0]  # Azul en BGR
    
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2[:, :] = [0, 255, 0]  # Verde en BGR
    
    img3 = np.zeros((100, 100, 3), dtype=np.uint8)
    img3[:, :] = [0, 0, 255]  # Rojo en BGR
    
    img4 = np.zeros((100, 100, 3), dtype=np.uint8)
    img4[:, :] = [255, 255, 0]  # Cian en BGR
    
    # Generar cuadrícula 2x2
    images = [img1, img2, img3, img4]
    labels = ["Azul", "Verde", "Rojo", "Cian"]
    grid = generate_grid_view(images, labels, cols=2, cell_size=(100, 100))
    
    # Verificar dimensiones
    assert grid.shape == (200, 200, 3)  # 2 filas x 2 columnas de 100x100
    
    # Verificar que los colores están en las posiciones correctas
    # Tomamos puntos alejados de los posibles textos de etiquetas (centro de cada celda)
    # Esquina superior izquierda (Azul)
    assert np.array_equal(grid[50, 50], np.array([255, 0, 0]))
    
    # Esquina superior derecha (Verde)
    assert np.array_equal(grid[50, 150], np.array([0, 255, 0]))
    
    # Esquina inferior izquierda (Rojo)
    assert np.array_equal(grid[150, 50], np.array([0, 0, 255]))
    
    # Esquina inferior derecha (Cian)
    assert np.array_equal(grid[150, 150], np.array([255, 255, 0]))

if __name__ == "__main__":
    test_list_image_files()
    test_list_image_files_empty_dir()
    test_list_image_files_nonexistent_dir()
    test_save_detection_result()
    test_crop_lpp_region()
    test_generate_grid_view()
    print("Todos los tests de utilidades de imágenes pasaron correctamente.")
