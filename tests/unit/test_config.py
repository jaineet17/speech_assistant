import unittest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from src.config
from src.config import PROJECT_ROOT, MODELS_DIR, DATA_DIR

class TestConfig(unittest.TestCase):
    """Test the config module functionality."""
    
    def test_project_paths(self):
        """Test that project paths are correctly defined."""
        self.assertTrue(isinstance(PROJECT_ROOT, Path))
        self.assertTrue(isinstance(MODELS_DIR, Path))
        self.assertTrue(isinstance(DATA_DIR, Path))
        
    def test_basic_functionality(self):
        """Basic test to ensure unittest is working."""
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main() 