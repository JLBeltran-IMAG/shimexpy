#!/usr/bin/env python3

"""
Este script prueba la instalación del paquete ShimExPy utilizando pyproject.toml.
Para usar:
    1. Instalar en modo desarrollo: pip install -e .
    2. Ejecutar este script: python test_install.py
"""

import sys
import importlib.util

# Lista de módulos a verificar
modules_to_check = [
    # Paquete principal
    "shimexpy",
    
    # Módulos del núcleo (core)
    "shimexpy.core",
    "shimexpy.core.spatial_harmonics",
    "shimexpy.core.contrast",
    "shimexpy.core.unwrapping",
    
    # Módulos de entrada/salida (io)
    "shimexpy.io",
    "shimexpy.io.file_io",
    
    # Módulos de utilidades
    "shimexpy.utils",
    "shimexpy.utils.crop",
    
    # Módulos de visualización
    "shimexpy.visualization",
    "shimexpy.visualization.plot",
    
    # Módulos de la GUI
    "shimexpy.gui",
    "shimexpy.gui.app",
    "shimexpy.gui.image_processor",
    "shimexpy.gui.image_widget",
    "shimexpy.gui.shimexpy_gui",
]

# Categorías de módulos para un mejor reporte
categories = {
    "core": [m for m in modules_to_check if ".core" in m],
    "io": [m for m in modules_to_check if ".io" in m],
    "utils": [m for m in modules_to_check if ".utils" in m],
    "visualization": [m for m in modules_to_check if ".visualization" in m],
    "gui": [m for m in modules_to_check if ".gui" in m],
    "main": ["shimexpy"],
}

# Asegurarse de que todos los módulos estén en alguna categoría
all_modules_in_categories = set()
for modules in categories.values():
    all_modules_in_categories.update(modules)

# Verificar si falta algún módulo en las categorías
for module in modules_to_check:
    if module not in all_modules_in_categories:
        print(f"Advertencia: El módulo {module} no está asignado a ninguna categoría.")

# Resultados por categoría
results = {cat: {"total": len(mods), "found": 0, "missing": []} for cat, mods in categories.items()}

# Verificar cada módulo
all_modules_present = True
print("\nVerificando instalación de ShimExPy...\n")

for module_name in modules_to_check:
    # Determinar a qué categoría pertenece este módulo
    category = None
    for cat, modules in categories.items():
        if module_name in modules:
            category = cat
            break
    if category is None:
        category = "main"  # Por defecto, si no encuentra categoría
    
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"[ERROR] Módulo {module_name} no encontrado")
            all_modules_present = False
            results[category]["missing"].append(module_name)
        else:
            print(f"[OK] Módulo {module_name} encontrado")
            results[category]["found"] += 1
            
    except ModuleNotFoundError:
        print(f"[ERROR] Módulo {module_name} no encontrado")
        all_modules_present = False
        results[category]["missing"].append(module_name)

# Imprimir resumen por categoría
print("\n" + "="*50)
print("RESUMEN DE INSTALACIÓN")
print("="*50)

for cat, res in results.items():
    status = "[OK] COMPLETO" if res["found"] == res["total"] else "[ERROR] INCOMPLETO"
    print(f"{cat.upper()}: {res['found']}/{res['total']} módulos encontrados - {status}")
    if res["missing"]:
        print(f"  Módulos faltantes: {', '.join(res['missing'])}")

if all_modules_present:
    print("\n[OK] ¡Instalación exitosa! Todos los módulos fueron encontrados.")
    print("Puedes ejecutar la aplicación con el comando: shimexpy")
else:
    print("\n[ERROR] Algunos módulos no pudieron ser encontrados. Revisa la instalación.")
    sys.exit(1)

# Intentar importar algunos módulos clave para verificar que no solo existen sino que funcionan
print("\nProbando importación de módulos clave...")
try:
    import shimexpy
    print("[OK] Importación básica funciona correctamente")
except Exception as e:
    print(f"[ERROR] Error al importar shimexpy: {str(e)}")
    all_modules_present = False
