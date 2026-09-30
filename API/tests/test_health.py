"""Unit tests for Health Check endpoint."""

import unittest
from fastapi.testclient import TestClient

from API.api import app


class HealthEndpointTests(unittest.TestCase):
    def test_health_check_returns_ok(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertTrue(data["model_loaded"])
            self.assertIn("version", data)


if __name__ == "__main__":
    unittest.main()
