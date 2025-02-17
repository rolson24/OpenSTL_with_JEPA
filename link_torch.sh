#!/bin/bash

# Directory containing the global torch installations
GLOBAL_TORCH_DIR="/home/bwct/.local/lib/python3.10/site-packages"

# Directory where the virtual environment packages are located
VENV_SITE_PACKAGES="venv/lib/python3.10/site-packages"

# Loop through all subdirectories in the global torch directory that start with "torch"
for dir in "$GLOBAL_TORCH_DIR"/torch*; do
    # Get the base name of the directory
    base_dir=$(basename "$dir")
    
    # Remove the old subdirectory in the virtual environment site-packages if it exists
    if [ -d "$VENV_SITE_PACKAGES/$base_dir" ] || [ -L "$VENV_SITE_PACKAGES/$base_dir" ]; then
        rm -rf "$VENV_SITE_PACKAGES/$base_dir"
    fi
    
    # Create a symbolic link in the virtual environment site-packages
    ln -s "$dir" "$VENV_SITE_PACKAGES/$base_dir"
done

echo "Soft links created successfully."