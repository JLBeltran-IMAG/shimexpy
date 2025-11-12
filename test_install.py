#!/usr/bin/env python3

"""
This script tests the installation of the ShimExPy package using pyproject.toml.
Usage:
    1. Install in development mode: pip install -e .
    2. Run this script: python test_install.py
"""

import sys
import importlib.util

# List of modules to check
modules_to_check = [
    # Main package
    "shimexpy",
    
    # Core modules
    "shimexpy.core",
    "shimexpy.core.spatial_harmonics",
    "shimexpy.core.contrast",
    "shimexpy.core.unwrapping",
    
    # IO modules
    "shimexpy.io",
    "shimexpy.io.file_io",
    
    # Utility modules
    "shimexpy.utils",
    "shimexpy.utils.crop",
    
    # Visualization modules
    "shimexpy.visualization",
    "shimexpy.visualization.plot",
    
    # GUI modules
    "shimexpy.gui",
    "shimexpy.gui.app",
    "shimexpy.gui.image_processor",
    "shimexpy.gui.image_widget",
    "shimexpy.gui.shimexpy_gui",
]

# Module categories for better reporting
categories = {
    "core": [m for m in modules_to_check if ".core" in m],
    "io": [m for m in modules_to_check if ".io" in m],
    "utils": [m for m in modules_to_check if ".utils" in m],
    "visualization": [m for m in modules_to_check if ".visualization" in m],
    "gui": [m for m in modules_to_check if ".gui" in m],
    "main": ["shimexpy"],
}

# Ensure all modules are in some category
all_modules_in_categories = set()
for modules in categories.values():
    all_modules_in_categories.update(modules)

# Check if any module is missing from categories
for module in modules_to_check:
    if module not in all_modules_in_categories:
        print(f"Warning: Module {module} is not assigned to any category.")

# Results by category
results = {cat: {"total": len(mods), "found": 0, "missing": []} for cat, mods in categories.items()}

# Check each module
all_modules_present = True
print("\nChecking ShimExPy installation...\n")

for module_name in modules_to_check:
    # Determine which category this module belongs to
    category = None
    for cat, modules in categories.items():
        if module_name in modules:
            category = cat
            break
    if category is None:
        category = "main"  # Default if no category found
    
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"[ERROR] Module {module_name} not found")
            all_modules_present = False
            results[category]["missing"].append(module_name)
        else:
            print(f"[OK] Module {module_name} found")
            results[category]["found"] += 1
            
    except ModuleNotFoundError:
        print(f"[ERROR] Module {module_name} not found")
        all_modules_present = False
        results[category]["missing"].append(module_name)

# Print summary by category
print("\n" + "="*50)
print("INSTALLATION SUMMARY")
print("="*50)

for cat, res in results.items():
    status = "[OK] COMPLETE" if res["found"] == res["total"] else "[ERROR] INCOMPLETE"
    print(f"{cat.upper()}: {res['found']}/{res['total']} modules found - {status}")
    if res["missing"]:
        print(f"  Missing modules: {', '.join(res['missing'])}")

if all_modules_present:
    print("\n[OK] Successful installation! All modules were found.")
    print("You can run the application with the command: shimexpy")
else:
    print("\n[ERROR] Some modules could not be found. Please check the installation.")
    sys.exit(1)

# Try importing some key modules to verify they not only exist but also work
print("\nTesting import of key modules...")
try:
    import shimexpy
    print("[OK] Basic import works correctly")
except Exception as e:
    print(f"[ERROR] Error importing shimexpy: {str(e)}")
    all_modules_present = False
