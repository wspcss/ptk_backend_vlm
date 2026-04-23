#!/bin/bash
# Navigate to the project directory
cd ~/Documents/vivi || { echo "Directory not found"; exit 1; }

# Execute the start script
echo "Launching Vivi services..."
./start_all.sh
