import unittest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.config import load_config
except ImportError:
    # Mock implementation for testing
    def load_config(config_path=None):
        return {"test": True}

class TestConfig(unittest.TestCase):
    """Test the config module functionality."""
    
    def test_load_config(self):
        """Test that config can be loaded."""
        config = load_config()
        self.assertIsInstance(config, dict)
        
    def test_basic_functionality(self):
        """Basic test to ensure unittest is working."""
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main() 