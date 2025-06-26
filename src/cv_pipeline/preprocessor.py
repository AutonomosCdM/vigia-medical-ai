"""
Preprocesador de imágenes para detección de lesiones por presión.

Este módulo implementa el preprocesamiento necesario para optimizar
las imágenes antes de enviarlas al detector de LPP.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image, ExifTags
import PIL

# Configuración de logging
logger = logging.getLogger('lpp-detect.preprocessor')

class ImagePreprocessor:
    """
    Preprocesador de imágenes para optimizar la detección de LPP.
    
    Implementa transformaciones como:
    - Redimensionamiento
    - Normalización
    - Eliminación de metadatos EXIF
    - Detección facial para enmascaramiento
    - Mejora de contraste para identificar eritemas
    """
    
    def __init__(self, target_size=(640, 640), normalize=True, face_detection=True,
                enhance_contrast=True, remove_exif=True):
        """
        Inicializa el preprocesador.
        
        Args:
            target_size: Dimensiones objetivo (ancho, alto)
            normalize: Normalizar valores de píxeles (0-1)
            face_detection: Activar detección facial y enmascaramiento
            enhance_contrast: Mejorar contraste para identificar eritemas
            remove_exif: Eliminar metadatos EXIF con información privada
        """
        self.target_size = target_size
        self.normalize = normalize
        self.face_detection = face_detection
        self.enhance_contrast = enhance_contrast
        self.remove_exif = remove_exif
        
        # Cargar detector facial si se solicita
        if self.face_detection:
            self._init_face_detector()
    
    def _init_face_detector(self):
        """Inicializa el detector facial."""
        try:
            # Usar detector facial de OpenCV (Haar Cascade)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info("Detector facial inicializado correctamente")
        except Exception as e:
            logger.warning(f"No se pudo inicializar detector facial: {str(e)}")
            self.face_detection = False
    
    def _remove_exif_data(self, pil_image):
        """Elimina metadatos EXIF que pueden contener información privada."""
        data = list(pil_image.getdata())
        image_without_exif = Image.new(pil_image.mode, pil_image.size)
        image_without_exif.putdata(data)
        return image_without_exif
    
    def _detect_and_blur_faces(self, cv_image):
        """Detecta rostros en la imagen y los difumina para proteger privacidad."""
        if not self.face_detection or self.face_detector is None:
            return cv_image
        
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Difuminar rostros detectados
        for (x, y, w, h) in faces:
            # Aplicar gaussianBlur al área del rostro
            face_roi = cv_image[y:y+h, x:x+w]
            cv_image[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi, (99, 99), 30)
            logger.info(f"Rostro detectado y difuminado en coordenadas: ({x}, {y}, {w}, {h})")
        
        return cv_image
    
    def _enhance_image_contrast(self, cv_image):
        """Mejora el contraste para mejor visualización de eritemas."""
        if not self.enhance_contrast:
            return cv_image
        
        # Convertir a espacio de color LAB (L=luminosidad, A=verde-rojo, B=azul-amarillo)
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        
        # Separar canales
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization) al canal L
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Mejorar canal A (verde-rojo) para destacar eritemas
        # Los eritemas tienen mayor componente roja, que está en canal A
        ca = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ca = ca.apply(a)
        
        # Combinar canales
        limg = cv2.merge((cl, ca, b))
        
        # Convertir de vuelta a BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def preprocess(self, image_path):
        """
        Preprocesa una imagen para optimizar la detección de LPP.
        
        Args:
            image_path: Ruta a la imagen o array NumPy
            
        Returns:
            numpy.ndarray: Imagen preprocesada como array NumPy
        """
        try:
            # Cargar imagen
            if isinstance(image_path, (str, Path)):
                # Cargar con PIL para manejar EXIF
                pil_image = Image.open(image_path)
                
                # Eliminar EXIF si se solicita
                if self.remove_exif:
                    pil_image = self._remove_exif_data(pil_image)
                
                # Convertir a array numpy
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                # Asumir que es un array numpy
                cv_image = image_path.copy()
            
            # Detectar y difuminar rostros
            if self.face_detection:
                cv_image = self._detect_and_blur_faces(cv_image)
            
            # Mejorar contraste para detectar eritemas
            if self.enhance_contrast:
                cv_image = self._enhance_image_contrast(cv_image)
            
            # Redimensionar
            cv_image = cv2.resize(cv_image, self.target_size)
            
            # Normalizar valores de píxeles si se solicita
            if self.normalize:
                cv_image = cv_image.astype(np.float32) / 255.0
            
            return cv_image
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            raise
    
    def get_preprocessor_info(self):
        """Retorna información sobre la configuración del preprocesador."""
        return {
            "target_size": self.target_size,
            "normalize": self.normalize,
            "face_detection": self.face_detection,
            "enhance_contrast": self.enhance_contrast,
            "remove_exif": self.remove_exif
        }
