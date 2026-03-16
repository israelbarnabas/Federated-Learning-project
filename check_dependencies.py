"""
Check that all required dependencies exist and are importable.
"""

import sys
import importlib

def check_module(module_name: str, critical: bool = True) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"  ✓ {module_name}")
        return True
    except ImportError as e:
        symbol = "✗" if critical else "⚠"
        print(f"  {symbol} {module_name}: {e}")
        return False

def main():
    print("Checking FL System Dependencies")
    print("=" * 60)
    
    critical_deps = [
        "flwr",
        "tensorflow",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
    ]
    
    optional_deps = [
        "rich",
        "tqdm",
    ]
    
    print("\nCritical dependencies:")
    critical_ok = all(check_module(m, critical=True) for m in critical_deps)
    
    print("\nOptional dependencies:")
    optional_ok = all(check_module(m, critical=False) for m in optional_deps)
    
    print("\n" + "=" * 60)
    
    # Check project files
    print("\nChecking project files:")
    required_files = [
        "try_project/task.py",
        "try_project/dp_utils.py",
        "try_project/enhanced_network_sim.py",
        "try_project/adaptive_scheduler.py",
        "try_project/enhanced_secure_agg.py",
        "try_project/enhanced_network_wrapper.py",
        "try_project/enhanced_client_app.py",
        "try_project/enhanced_server_app.py",
    ]
    
    import os
    files_ok = True
    for f in required_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} MISSING")
            files_ok = False
    
    print("\n" + "=" * 60)
    if critical_ok and files_ok:
        print("✓ All critical dependencies and files present")
        return 0
    else:
        print("✗ Missing critical dependencies or files")
        return 1

if __name__ == "__main__":
    sys.exit(main())