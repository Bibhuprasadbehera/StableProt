import unittest
import sys
import os

if __name__ == '__main__':
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("="*80)
    print("  STABLEPROT V2 - MASTER TEST SUITE")
    print("="*80)
    
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*80)
    if result.wasSuccessful():
        print("  ✅ ALL TESTS PASSED. Pipeline integrity verified.")
        sys.exit(0)
    else:
        print("  ❌ TESTS FAILED. Please review the errors above.")
        sys.exit(1)
