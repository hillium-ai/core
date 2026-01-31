import unittest
from unittest.mock import MagicMock

from loqus_core.inference import PowerInferBackend, GenerateParams, GenerateResult, get_backend


class TestPowerInferBackend(unittest.TestCase):

    def test_backend_creation(self):
        backend = PowerInferBackend()
        self.assertIsInstance(backend, PowerInferBackend)

    def test_get_backend_factory(self):
        backend = get_backend("powerinfer")
        self.assertIsInstance(backend, PowerInferBackend)

    def test_load_model_with_ffi(self):
        mock_pyo3 = MagicMock()
        mock_pyo3.powerinfer_load_model.return_value = "mock_handle_123"
        mock_pyo3.powerinfer_is_loaded.return_value = True

        backend = PowerInferBackend()
        backend._ffi_module = mock_pyo3

        backend.load_model("/fake/model.gguf", {})
        self.assertTrue(backend.is_loaded())
        mock_pyo3.powerinfer_load_model.assert_called_once_with("/fake/model.gguf", {})

    def test_load_model_without_ffi(self):
        backend = PowerInferBackend()
        backend._ffi_module = None
        backend.load_model("/fake/model.gguf", {})
        self.assertTrue(backend.is_loaded())

    def test_generate_with_ffi(self):
        mock_pyo3 = MagicMock()
        mock_result = '{"text": "Hello world!", "tokens_generated": 2, "latency_ms": 5.0, "finish_reason": "stop"}'
        mock_pyo3.powerinfer_generate.return_value = mock_result
        mock_pyo3.powerinfer_is_loaded.return_value = True

        backend = PowerInferBackend()
        backend._model_handle = "mock_handle_123"
        backend._ffi_module = mock_pyo3
        backend._is_loaded = True

        result = backend.generate("Hello", GenerateParams())
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result.text, "Hello world!")

    def test_generate_without_ffi(self):
        backend = PowerInferBackend()
        backend._ffi_module = None
        backend._is_loaded = True

        result = backend.generate("Hello", GenerateParams())
        self.assertIsInstance(result, GenerateResult)
        self.assertIn("[MOCK]", result.text)

    def test_unload(self):
        backend = PowerInferBackend()
        backend.unload()
        self.assertFalse(backend.is_loaded())

    def test_is_loaded_default(self):
        backend = PowerInferBackend()
        backend._ffi_module = None  # force mock mode
        self.assertFalse(backend.is_loaded())

    def test_is_loaded_after_mock_load(self):
        backend = PowerInferBackend()
        backend._ffi_module = None
        backend.load_model("/fake/model.gguf", {})
        self.assertTrue(backend.is_loaded())

    def test_generate_result_dataclass(self):
        r = GenerateResult(text="hi", tokens_generated=1, latency_ms=1.0, finish_reason="stop")
        self.assertEqual(r.text, "hi")
        self.assertEqual(r.tokens_generated, 1)


if __name__ == '__main__':
    unittest.main()
