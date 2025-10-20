#!/usr/bin/env python3
"""
Test script to verify  work correctly.
Tests core functionality without requiring GPU dependencies.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def test_carbon_calculator():
    """Test simplified carbon calculator."""
    try:
        from measurement.carbon_calculator import CarbonCalculator, CarbonIntensityConfig
        print("Carbon calculator import: OK")
        
        # Test default config
        config = CarbonIntensityConfig()
        print(f" Default carbon intensity: {config.intensity_gco2e_per_kwh} gCO2e/kWh")
        
        # Test calculator
        calc = CarbonCalculator()
        result = calc.calculate_carbon_footprint(1000.0)  # 1000 Joules
        print(f"Carbon calculation test: {result:.6f} gCO2e")
        
        return True
    except Exception as e:
        print(f" Carbon calculator error: {e}")
        return False

def test_experiment_config():
    """Test experiment configuration loading."""
    try:
        import yaml
        
        # Test loading minimal config
        config_path = Path("configs/experiment.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print("Experiment config loading: OK")
            print(f" Config name: {config['experiment']['name']}")
            print(f" Repetitions: {config['design']['repetitions']}")
            return True
        else:
            print(" Config file not found")
            return False
            
    except Exception as e:
        print(f" Config loading error: {e}")
        return False

def test_task_definitions():
    """Test task definitions work."""
    try:
        # Direct import to avoid package issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "task_definitions", 
            src_path / "tasks" / "task_definitions.py"
        )
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        
        print("Task definitions import: OK")
        
        # Test basic functionality
        TaskCategory = task_module.TaskCategory
        print(f" Task categories: {[cat.value for cat in TaskCategory]}")
        
        return True
    except Exception as e:
        print(f"Task definitions error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing simplified codebase functionality...\n")
    
    tests = [
        ("Carbon Calculator", test_carbon_calculator),
        ("Experiment Config", test_experiment_config), 
        ("Task Definitions", test_task_definitions),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}:")
        success = test_func()
        results.append(success)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All core functionality working after simplifications")
    else:
        print(" Some functionality broken by simplifications")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)