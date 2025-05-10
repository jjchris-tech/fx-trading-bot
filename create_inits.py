"""
Create Init Files
Creates all necessary __init__.py files for the project structure.
"""
import os
from pathlib import Path

def create_init_files():
    """Create __init__.py files in all module directories."""
    # Define the module structure
    module_structure = {
        "config": [],
        "data": [],
        "execution": [],
        "strategies": [],
        "optimization": [],
        "reporting": [],
        "alerts": [],
        "utils": [],
        "tests": []
    }
    
    # Create directories and __init__.py files
    for module, submodules in module_structure.items():
        # Create module directory if it doesn't exist
        module_dir = Path(module)
        module_dir.mkdir(exist_ok=True)
        
        # Create __init__.py in module directory
        init_file = module_dir / "__init__.py"
        if not init_file.exists():
            with open(init_file, "w") as f:
                f.write(f'"""\n{module.capitalize()} Module\n"""\n')
            print(f"Created {init_file}")
        
        # Create submodules and their __init__.py files
        for submodule in submodules:
            submodule_dir = module_dir / submodule
            submodule_dir.mkdir(exist_ok=True)
            
            sub_init_file = submodule_dir / "__init__.py"
            if not sub_init_file.exists():
                with open(sub_init_file, "w") as f:
                    f.write(f'"""\n{submodule.capitalize()} Submodule\n"""\n')
                print(f"Created {sub_init_file}")

if __name__ == "__main__":
    create_init_files()
    print("All __init__.py files created successfully.")