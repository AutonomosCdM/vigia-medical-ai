"""
Sistemas Especializados - Capa 3
Sistemas de procesamiento con conocimiento médico profundo.
"""

# Temporarily skip problematic imports for medical testing
try:
    from .clinical_processing import ClinicalProcessingSystem, ClinicalProcessingFactory
    __all__ = ["ClinicalProcessingSystem", "ClinicalProcessingFactory"]
except ImportError:
    # Skip if dependencies not available
    __all__ = []