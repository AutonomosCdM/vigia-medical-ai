"""
Fixtures compartidas para todos los tests del proyecto Vigía.
Centraliza la configuración de tests y elimina duplicación.
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from PIL import Image
from datetime import datetime

# Agregar el proyecto al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== FIXTURES DE CONFIGURACIÓN ====================

@pytest.fixture(scope="session")
def test_env_vars():
    """Variables de entorno para tests"""
    return {
        'SUPABASE_URL': 'https://test.supabase.co',
        'SUPABASE_KEY': 'test-key-123',
        'TWILIO_ACCOUNT_SID': 'ACtest123',
        'TWILIO_AUTH_TOKEN': 'test-token-456',
        'TWILIO_WHATSAPP_FROM': 'whatsapp:+14155238886',
        'SLACK_BOT_TOKEN': 'xoxb-test-token',
        'SLACK_SIGNING_SECRET': 'test-signing-secret',
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': '6379',
        'ENVIRONMENT': 'test'
    }


@pytest.fixture(autouse=True)
def setup_test_env(test_env_vars, monkeypatch):
    """Configura variables de entorno para todos los tests"""
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def temp_dir():
    """Directorio temporal para tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==================== FIXTURES DE DATOS ====================

@pytest.fixture
def sample_patient_data():
    """Datos de paciente de ejemplo"""
    return {
        'patient_code': 'TC-2025-001',
        'name': 'Test Case',
        'age': 65,
        'service': 'Test Service',
        'bed': '101-A',
        'diagnoses': ['Test diagnosis 1', 'Test diagnosis 2'],
        'medications': ['Test med 1', 'Test med 2'],
        'risk_score': 15
    }


@pytest.fixture
def sample_detection_result():
    """Resultado de detección de ejemplo"""
    return {
        'detections': [
            {
                'class': 2,
                'confidence': 0.92,
                'bbox': [100, 100, 200, 200],
                'location': 'Sacro'
            }
        ],
        'inference_time': 0.123,
        'preprocessing_time': 0.045,
        'total_time': 0.168,
        'max_severity': 2
    }


@pytest.fixture
def sample_whatsapp_message():
    """Mensaje de WhatsApp de ejemplo"""
    return {
        'From': 'whatsapp:+56912345678',
        'To': 'whatsapp:+14155238886',
        'Body': 'Hola, envío foto del paciente César Durán',
        'MessageSid': 'SMtest123',
        'AccountSid': 'ACtest123',
        'NumMedia': '1',
        'MediaUrl0': 'https://api.twilio.com/test-image.jpg',
        'MediaContentType0': 'image/jpeg'
    }


@pytest.fixture
def sample_slack_event():
    """Evento de Slack de ejemplo"""
    return {
        'type': 'block_actions',
        'user': {'id': 'U123456', 'name': 'test_user'},
        'channel': {'id': 'C123456', 'name': 'test_channel'},
        'trigger_id': 'test_trigger_123',
        'actions': [{
            'action_id': 'ver_historial_medico',
            'block_id': 'block_123',
            'value': 'CD-2025-001',
            'type': 'button'
        }]
    }


# ==================== FIXTURES DE IMÁGENES ====================

@pytest.fixture
def sample_image(temp_dir):
    """Crea una imagen de prueba"""
    # Crear imagen RGB de 640x480
    img = Image.new('RGB', (640, 480), color='white')
    
    # Agregar un rectángulo rojo (simulando lesión)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 150, 400, 300], fill='red')
    
    # Guardar imagen
    image_path = temp_dir / 'test_image.jpg'
    img.save(image_path)
    
    return str(image_path)


@pytest.fixture
def sample_image_array():
    """Array numpy de imagen de prueba"""
    # Imagen RGB 640x480
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Agregar región roja (simulando lesión)
    img[150:300, 200:400] = [255, 0, 0]
    
    return img


@pytest.fixture
def batch_images(temp_dir):
    """Crea múltiples imágenes de prueba"""
    images = []
    for i in range(5):
        img = Image.new('RGB', (640, 480), color='white')
        draw = ImageDraw.Draw(img)
        # Diferentes posiciones para cada imagen
        x = 100 + i * 50
        y = 100 + i * 30
        draw.rectangle([x, y, x+100, y+100], fill='red')
        
        image_path = temp_dir / f'test_image_{i}.jpg'
        img.save(image_path)
        images.append(str(image_path))
    
    return images


# ==================== MOCKS DE SERVICIOS ====================

@pytest.fixture
def mock_supabase_client():
    """Mock del cliente Supabase"""
    client = MagicMock()
    
    # Mock de métodos comunes
    client.table.return_value.select.return_value.execute.return_value.data = []
    client.table.return_value.insert.return_value.execute.return_value.data = [
        {'id': 'test-id-123', 'created_at': '2025-01-01T00:00:00'}
    ]
    
    return client


@pytest.fixture
def mock_twilio_client():
    """Mock del cliente Twilio"""
    client = MagicMock()
    
    # Mock de envío de mensaje
    message = MagicMock()
    message.sid = 'SMtest123'
    message.status = 'sent'
    client.messages.create.return_value = message
    
    return client


@pytest.fixture
def mock_slack_client():
    """Mock del cliente Slack"""
    client = MagicMock()
    
    # Mock de métodos comunes
    client.chat_postMessage.return_value = {'ok': True, 'ts': '123.456'}
    client.views_open.return_value = {'ok': True}
    
    return client


@pytest.fixture
def mock_redis_client():
    """Mock del cliente Redis"""
    client = MagicMock()
    
    # Mock de métodos comunes
    client.get.return_value = None
    client.set.return_value = True
    client.exists.return_value = False
    
    return client


# ==================== FIXTURES DE MODELOS ====================

@pytest.fixture
def mock_yolo_model():
    """Mock del modelo YOLO"""
    model = MagicMock()
    
    # Mock de predicción
    result = MagicMock()
    result.xyxy = [torch.tensor([[100, 100, 200, 200, 0.92, 2]])]
    result.pandas.return_value.xyxy = [
        MagicMock(values=np.array([[100, 100, 200, 200, 0.92, 2]]))
    ]
    
    model.return_value = [result]
    
    return model


# ==================== FIXTURES DE CONTEXTO ====================

@pytest.fixture
def vigia_context(mock_supabase_client, mock_twilio_client, 
                  mock_slack_client, mock_redis_client):
    """Contexto completo para tests de integración"""
    return {
        'supabase': mock_supabase_client,
        'twilio': mock_twilio_client,
        'slack': mock_slack_client,
        'redis': mock_redis_client,
        'config': {
            'model_type': 'yolov5s',
            'confidence_threshold': 0.25,
            'environment': 'test'
        }
    }


# ==================== UTILIDADES ====================

@pytest.fixture
def capture_logs():
    """Captura logs durante tests"""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Agregar handler a todos los loggers de vigia
    loggers = [
        logging.getLogger('vigia'),
        logging.getLogger('vigia-detect'),
        logging.getLogger('vigia.supabase'),
        logging.getLogger('vigia.twilio'),
        logging.getLogger('vigia.slack')
    ]
    
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    # Limpiar handlers
    for logger in loggers:
        logger.removeHandler(handler)


@pytest.fixture
def assert_detection_valid():
    """Helper para validar detecciones"""
    def _assert(detection):
        assert 'class' in detection
        assert 'confidence' in detection
        assert 'bbox' in detection
        assert len(detection['bbox']) == 4
        assert 0 <= detection['class'] <= 4
        assert 0 <= detection['confidence'] <= 1
    
    return _assert


# ==================== CONFIGURACIÓN DE PYTEST ====================

def pytest_configure(config):
    """Configuración global de pytest"""
    # Agregar markers personalizados
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers", "slow: marca tests lentos"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marca tests que requieren GPU"
    )


# Importar torch solo si está disponible (para el mock de YOLO)
try:
    import torch
except ImportError:
    torch = None


# ==================== EJEMPLOS DE USO ====================
"""
Ejemplos de cómo usar estas fixtures en tests:

def test_process_image(sample_image, mock_yolo_model):
    # Test con imagen y modelo mockeado
    processor = ImageProcessor()
    result = processor.process_image(sample_image)
    assert result['success'] is True

def test_save_detection(sample_detection_result, mock_supabase_client):
    # Test guardando en BD mockeada
    client = SupabaseClient()
    client.client = mock_supabase_client
    result = client.save_detection('patient-123', sample_detection_result)
    assert result is not None

def test_integration(vigia_context, sample_image):
    # Test de integración con contexto completo
    # Todos los servicios externos están mockeados
    pass
"""