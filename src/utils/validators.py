"""
Validadores centralizados para el sistema Vigía.
Elimina duplicación de validaciones en múltiples módulos.
"""
import re
import os
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path


class PhoneValidator:
    """Validaciones para números telefónicos"""
    
    @staticmethod
    def validate_phone_number(number: str) -> bool:
        """
        Valida que un número telefónico tenga formato internacional válido.
        
        Args:
            number: Número a validar
            
        Returns:
            True si es válido, False en caso contrario
        """
        if not number:
            return False
        
        # Remover prefijo whatsapp: si existe
        clean_number = number.replace('whatsapp:', '').strip()
        
        # Verificar formato internacional (+XX...)
        if not clean_number.startswith('+'):
            return False
        
        # Remover el + y verificar que solo tenga dígitos
        digits_only = clean_number[1:].replace(' ', '').replace('-', '')
        
        # Debe tener al menos 10 dígitos y máximo 15 (estándar E.164)
        return digits_only.isdigit() and 10 <= len(digits_only) <= 15
    
    @staticmethod
    def format_whatsapp_number(number: str) -> str:
        """
        Formatea un número para WhatsApp.
        
        Args:
            number: Número a formatear
            
        Returns:
            Número formateado con prefijo whatsapp:
        """
        if not number:
            return number
        
        # Si ya tiene el prefijo, retornarlo tal cual
        if number.startswith('whatsapp:'):
            return number
        
        # Agregar prefijo
        return f'whatsapp:{number}'
    
    @staticmethod
    def format_chilean_number(number: str) -> str:
        """
        Formatea un número chileno al formato internacional.
        
        Args:
            number: Número en formato local (9XXXXXXXX) o internacional
            
        Returns:
            Número en formato internacional (+569XXXXXXXX)
        """
        # Remover espacios y guiones
        clean = number.replace(' ', '').replace('-', '')
        
        # Si ya es internacional, retornarlo
        if clean.startswith('+56'):
            return clean
        
        # Si empieza con 56, agregar +
        if clean.startswith('56'):
            return f'+{clean}'
        
        # Si es número móvil chileno (9 dígitos empezando con 9)
        if clean.startswith('9') and len(clean) == 9:
            return f'+56{clean}'
        
        # Si no se puede formatear, retornar original
        return number
    
    @staticmethod
    def extract_country_code(number: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrae el código de país de un número internacional.
        
        Args:
            number: Número internacional
            
        Returns:
            Tupla (código_país, número_sin_código)
        """
        clean = number.replace('whatsapp:', '').strip()
        
        if not clean.startswith('+'):
            return None, clean
        
        # Códigos de país comunes (expandir según necesidad)
        country_codes = {
            '1': 'US/CA',      # USA/Canadá
            '52': 'MX',        # México
            '54': 'AR',        # Argentina
            '55': 'BR',        # Brasil
            '56': 'CL',        # Chile
            '57': 'CO',        # Colombia
            '58': 'VE',        # Venezuela
            '593': 'EC',       # Ecuador
            '595': 'PY',       # Paraguay
            '598': 'UY',       # Uruguay
        }
        
        # Buscar coincidencia de código
        for code, country in country_codes.items():
            if clean[1:].startswith(code):
                return country, clean[1+len(code):]
        
        return 'Unknown', clean[1:]


class ImageValidator:
    """Validaciones para archivos de imagen"""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for medical images
    
    @staticmethod
    def is_valid_image(file_path: str) -> bool:
        """
        Valida que un archivo sea una imagen válida.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            True si es imagen válida, False en caso contrario
        """
        if not file_path or not os.path.exists(file_path):
            return False
        
        path = Path(file_path)
        
        # Verificar extensión
        if path.suffix.lower() not in ImageValidator.VALID_EXTENSIONS:
            return False
        
        # Verificar tamaño
        try:
            file_size = path.stat().st_size
            if file_size > ImageValidator.MAX_FILE_SIZE:
                return False
        except:
            return False
        
        return True
    
    @staticmethod
    def validate_image_dimensions(width: int, height: int, 
                                 min_size: int = 100,
                                 max_size: int = 4096) -> Tuple[bool, Optional[str]]:
        """
        Valida las dimensiones de una imagen.
        
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            min_size: Tamaño mínimo permitido
            max_size: Tamaño máximo permitido
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if width < min_size or height < min_size:
            return False, f"Imagen muy pequeña. Mínimo {min_size}x{min_size}"
        
        if width > max_size or height > max_size:
            return False, f"Imagen muy grande. Máximo {max_size}x{max_size}"
        
        # Verificar aspect ratio (evitar imágenes muy distorsionadas)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 10:
            return False, "Relación de aspecto inválida (muy alargada)"
        
        return True, None


class PatientValidator:
    """Validaciones para datos de pacientes"""
    
    @staticmethod
    def validate_patient_code(code: str) -> bool:
        """
        Valida el formato del código de paciente.
        Formato esperado: XX-YYYY-NNN (ej: CD-2025-001)
        
        Args:
            code: Código a validar
            
        Returns:
            True si es válido
        """
        if not code:
            return False
        
        # Patrón: 2 letras - 4 dígitos - 3 dígitos
        pattern = r'^[A-Z]{2}-\d{4}-\d{3}$'
        return bool(re.match(pattern, code))
    
    @staticmethod
    def generate_patient_code(initials: str, year: Optional[int] = None) -> str:
        """
        Genera un código de paciente.
        
        Args:
            initials: Iniciales del paciente (2 letras)
            year: Año (por defecto año actual)
            
        Returns:
            Código generado
        """
        from datetime import datetime
        
        # Validar y formatear iniciales
        initials = initials.upper()[:2].ljust(2, 'X')
        
        # Año
        if not year:
            year = datetime.now().year
        
        # TODO: En producción, obtener el siguiente número de la BD
        # Por ahora usar timestamp
        number = str(datetime.now().timestamp())[-3:]
        
        return f"{initials}-{year}-{number}"
    
    @staticmethod
    def validate_age(age: int) -> Tuple[bool, Optional[str]]:
        """
        Valida la edad del paciente.
        
        Args:
            age: Edad en años
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if age < 0:
            return False, "La edad no puede ser negativa"
        
        if age > 120:
            return False, "Edad fuera de rango válido"
        
        if age < 18:
            return True, "Paciente pediátrico - requiere consentimiento del tutor"
        
        return True, None


def validate_patient_code_format(code: str) -> bool:
    """
    Valida formato de código de paciente.
    
    Args:
        code: Código a validar
        
    Returns:
        True si el formato es válido
    """
    if not code:
        return False
    
    # Formato: 2-3 letras seguidas de guión y números/año
    pattern = r'^[A-Z]{2,3}-\d{4}-\d{3,4}$'
    return bool(re.match(pattern, code))


class ClinicalValidator:
    """Validaciones para datos clínicos"""
    
    @staticmethod
    def validate_lpp_grade(grade: int) -> bool:
        """
        Valida que el grado de LPP sea válido (0-4).
        
        Args:
            grade: Grado a validar
            
        Returns:
            True si es válido
        """
        return isinstance(grade, int) and 0 <= grade <= 4
    
    @staticmethod
    def validate_confidence_score(score: float) -> bool:
        """
        Valida que un score de confianza esté en rango válido (0-1).
        
        Args:
            score: Score a validar
            
        Returns:
            True si es válido
        """
        return isinstance(score, (int, float)) and 0 <= score <= 1
    
    @staticmethod
    def validate_anatomical_location(location: str) -> Tuple[bool, Optional[str]]:
        """
        Valida que la ubicación anatómica sea reconocida.
        
        Args:
            location: Ubicación anatómica
            
        Returns:
            Tupla (es_válido, ubicación_normalizada)
        """
        valid_locations = {
            'sacro': 'Sacro',
            'sacrum': 'Sacro',
            'talon': 'Talón',
            'talón': 'Talón',
            'heel': 'Talón',
            'codo': 'Codo',
            'elbow': 'Codo',
            'escapula': 'Escápula',
            'scapula': 'Escápula',
            'trocanter': 'Trocánter',
            'trochanter': 'Trocánter',
            'occipital': 'Occipital',
            'isquion': 'Isquion',
            'ischium': 'Isquion'
        }
        
        location_lower = location.lower().strip()
        
        if location_lower in valid_locations:
            return True, valid_locations[location_lower]
        
        # Si no está en la lista, aceptarlo pero sin normalizar
        return True, location
    
    @staticmethod
    def validate_medication_dose(dose_string: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Valida y parsea una dosis de medicamento.
        
        Args:
            dose_string: String de dosis (ej: "850mg c/12h")
            
        Returns:
            Tupla (es_válido, datos_parseados)
        """
        # Patrón para dosis: número + unidad + frecuencia opcional
        pattern = r'^(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|ui)\s*(?:c/(\d+)h)?$'
        match = re.match(pattern, dose_string.lower())
        
        if not match:
            return False, None
        
        amount, unit, frequency = match.groups()
        
        parsed = {
            'amount': float(amount),
            'unit': unit,
            'frequency_hours': int(frequency) if frequency else None
        }
        
        return True, parsed


# Funciones de conveniencia para uso directo
def validate_phone(number: str) -> bool:
    """Valida un número telefónico"""
    return PhoneValidator.validate_phone_number(number)


def validate_image(path: str) -> bool:
    """Valida un archivo de imagen"""
    return ImageValidator.is_valid_image(path)


def validate_patient_code(code: str) -> bool:
    """Valida un código de paciente"""
    return PatientValidator.validate_patient_code(code)


def validate_lpp_grade(grade: int) -> bool:
    """Valida un grado de LPP"""
    return ClinicalValidator.validate_lpp_grade(grade)


if __name__ == "__main__":
    # Ejemplos de uso
    print("📱 Validación de teléfonos:")
    print(f"  +56912345678: {validate_phone('+56912345678')}")
    print(f"  912345678: {validate_phone('912345678')}")
    print(f"  Formato WhatsApp: {PhoneValidator.format_whatsapp_number('+56912345678')}")
    
    print("\n🏥 Validación de pacientes:")
    print(f"  CD-2025-001: {validate_patient_code('CD-2025-001')}")
    print(f"  Generar código: {PatientValidator.generate_patient_code('JD')}")
    
    print("\n💊 Validación clínica:")
    print(f"  Grado 2: {validate_lpp_grade(2)}")
    print(f"  Grado 5: {validate_lpp_grade(5)}")
    
    valid, parsed = ClinicalValidator.validate_medication_dose("850mg c/12h")
    print(f"  Dosis '850mg c/12h': {valid}, {parsed}")