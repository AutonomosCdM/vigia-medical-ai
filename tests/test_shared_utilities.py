"""
Tests for shared utilities.
"""
import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from src.utils.shared_utilities import (
    VigiaLogger,
    VigiaValidator, 
    VigiaErrorHandler,
    with_error_handling,
    with_validation,
    PerformanceTracker
)
from tests.shared_fixtures import (
    temp_directory,
    sample_image_path,
    invalid_patient_codes
)


class TestVigiaLogger:
    
    def test_get_logger_creates_new_logger(self):
        """Test that get_logger creates a new logger with correct name"""
        logger = VigiaLogger.get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "vigia.test_module"
    
    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same instance for same name"""
        logger1 = VigiaLogger.get_logger("test_module")
        logger2 = VigiaLogger.get_logger("test_module")
        
        assert logger1 is logger2
    
    def test_logger_has_handlers(self):
        """Test that logger has console handler configured"""
        logger = VigiaLogger.get_logger("test_handler")
        
        assert len(logger.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


class TestVigiaValidator:
    
    def test_validate_patient_code_valid(self):
        """Test validation of valid patient codes"""
        valid_codes = ["CD-2025-001", "AB-2024-999", "XY-2025-123"]
        
        for code in valid_codes:
            result = VigiaValidator.validate_patient_code(code)
            assert result["valid"] is True
            assert result["patient_code"] == code
    
    def test_validate_patient_code_invalid(self, invalid_patient_codes):
        """Test validation of invalid patient codes"""
        for code in invalid_patient_codes:
            result = VigiaValidator.validate_patient_code(code)
            assert result["valid"] is False
            assert "error" in result
    
    def test_validate_image_file_valid(self, sample_image_path):
        """Test validation of valid image file"""
        result = VigiaValidator.validate_image_file(sample_image_path)
        
        assert result["valid"] is True
        assert result["file_path"] == str(sample_image_path)
    
    def test_validate_image_file_not_exists(self):
        """Test validation of non-existent file"""
        result = VigiaValidator.validate_image_file("/nonexistent/file.jpg")
        
        assert result["valid"] is False
        assert "does not exist" in result["error"]
    
    def test_validate_image_file_invalid_extension(self, temp_directory):
        """Test validation of file with invalid extension"""
        invalid_file = temp_directory / "test.txt"
        invalid_file.write_text("test content")
        
        result = VigiaValidator.validate_image_file(invalid_file)
        
        assert result["valid"] is False
        assert "Invalid image format" in result["error"]
    
    def test_validate_detection_confidence_valid(self):
        """Test validation of valid confidence values"""
        valid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for value in valid_values:
            result = VigiaValidator.validate_detection_confidence(value)
            assert result["valid"] is True
            assert result["confidence"] == float(value)
    
    def test_validate_detection_confidence_invalid(self):
        """Test validation of invalid confidence values"""
        invalid_values = [-0.1, 1.1, "invalid", None]
        
        for value in invalid_values:
            result = VigiaValidator.validate_detection_confidence(value)
            assert result["valid"] is False
            assert "error" in result


class TestVigiaErrorHandler:
    
    def test_handle_exception(self):
        """Test exception handling"""
        logger = Mock()
        exception = ValueError("Test error")
        
        result = VigiaErrorHandler.handle_exception(
            logger, "test_operation", exception, {"key": "value"}
        )
        
        assert result["success"] is False
        assert "error" in result
        assert result["error"]["operation"] == "test_operation"
        assert result["error"]["error_type"] == "ValueError"
        assert result["error"]["error_message"] == "Test error"
        assert result["error"]["context"] == {"key": "value"}
        
        # Check that logger was called
        logger.error.assert_called()
        logger.debug.assert_called()
    
    def test_create_success_response(self):
        """Test success response creation"""
        data = {"result": "test_data"}
        operation = "test_operation"
        message = "Custom success message"
        
        result = VigiaErrorHandler.create_success_response(data, operation, message)
        
        assert result["success"] is True
        assert result["operation"] == operation
        assert result["data"] == data
        assert result["message"] == message
        assert "timestamp" in result


class TestErrorHandlingDecorator:
    
    def test_with_error_handling_success(self):
        """Test error handling decorator with successful function"""
        @with_error_handling("test_operation")
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        assert result == 5
    
    def test_with_error_handling_exception(self):
        """Test error handling decorator with exception"""
        @with_error_handling("test_operation")
        def test_function():
            raise ValueError("Test error")
        
        result = test_function()
        
        assert result["success"] is False
        assert "error" in result
        assert result["error"]["error_type"] == "ValueError"


class TestValidationDecorator:
    
    def test_with_validation_success(self):
        """Test validation decorator with successful validation"""
        def validator(*args, **kwargs):
            return {"valid": True}
        
        @with_validation([validator])
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        assert result == 10
    
    def test_with_validation_failure(self):
        """Test validation decorator with failed validation"""
        def validator(*args, **kwargs):
            return {"valid": False, "error": "Validation failed"}
        
        @with_validation([validator])
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        
        assert result["success"] is False
        assert result["error"] == "Validation failed"
    
    def test_with_validation_exception(self):
        """Test validation decorator with validator exception"""
        def validator(*args, **kwargs):
            raise Exception("Validator error")
        
        @with_validation([validator])
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        
        assert result["success"] is False
        assert "Validation error" in result["error"]


class TestPerformanceTracker:
    
    def test_performance_tracker_context_manager(self):
        """Test performance tracker as context manager"""
        with PerformanceTracker("test_operation") as tracker:
            # Simulate some work
            pass
        
        duration = tracker.get_duration()
        assert duration is not None
        assert duration >= 0
    
    def test_performance_tracker_with_exception(self):
        """Test performance tracker when exception occurs"""
        try:
            with PerformanceTracker("test_operation") as tracker:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        duration = tracker.get_duration()
        assert duration is not None
        assert duration >= 0
    
    def test_performance_tracker_get_duration_before_completion(self):
        """Test get_duration before context manager completion"""
        tracker = PerformanceTracker("test_operation")
        
        # Should return None before entering context
        assert tracker.get_duration() is None