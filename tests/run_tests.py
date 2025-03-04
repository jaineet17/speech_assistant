#!/usr/bin/env python3
"""Test runner for the speech assistant project."""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests(test_type="all"):
    """Run the specified tests.
    
    Args:
        test_type: Type of tests to run ("unit", "integration", or "all")
    
    Returns:
        True if all tests pass, False otherwise
    """
    loader = unittest.TestLoader()
    
    if test_type == "unit" or test_type == "all":
        print("Running unit tests...")
        unit_tests = loader.discover("tests/unit")
        unit_result = unittest.TextTestRunner(verbosity=2).run(unit_tests)
        
        if unit_result.failures or unit_result.errors:
            print("Unit tests failed!")
            if test_type == "unit":
                return False
        else:
            print("All unit tests passed!")
    
    if test_type == "integration" or test_type == "all":
        print("\nRunning integration tests...")
        integration_tests = loader.discover("tests/integration")
        integration_result = unittest.TextTestRunner(verbosity=2).run(integration_tests)
        
        if integration_result.failures or integration_result.errors:
            print("Integration tests failed!")
            return False
        else:
            print("All integration tests passed!")
    
    return True


if __name__ == "__main__":
    # Parse command line arguments
    test_type = "all"
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type not in ["unit", "integration", "all"]:
            print(f"Invalid test type: {test_type}")
            print("Valid options: unit, integration, all")
            sys.exit(1)
    
    # Run tests
    success = run_tests(test_type)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 