#!/bin/bash
# Ścieżka do pliku blokady
LOCK_FILE="/workspace/startup.lock"

# Sprawdź, czy skrypt już został uruchomiony
if [ -f "$LOCK_FILE" ]; then
    echo "Startup script already executed. Exiting."
    /bin/bash
fi

echo "Executing startup script..."
# Tworzenie pliku blokady
touch "$LOCK_FILE"

apt update && apt install -y screen
pip install -r requirements.txt

echo "Installed dependencies."

# Clone the main repo if it doesn't exist
# if [ ! -d "/workspace/core" ]; 
# then
#     echo "Cloning the main repo..."
#     git clone https://$gitLabLogin:$gitLabPassword@gitlab.d0d0.ovh/ai/architecture-tweak.git /workspace/temp
#     # Move the main repo to the correct directory
#     cd /workspace/temp
#     rm -rf core utils
#     mv /workspace/temp/* /workspace/.
#     rm -rf /workspace/temp
# else
#     echo " Repo exist, Pulling the main repo..."
#     cd /workspace && git pull
# fi

# Clone the latest data 
if [ -d "/workspace/data/mwoutput" ]; 
then
    rm -rf /workspace/data/*
fi

echo "Cloning the data repo..."
git clone https://$gitLabLogin:$gitLabPassword@gitlab.d0d0.ovh/ai/ai-handwriting.git /workspace/temp
# Move the data repo to the correct directory
mv /workspace/temp/mwoutput /workspace/data/.
mv /workspace/temp/output /workspace/data/.
rm -rf /workspace/temp

echo "Cloned the repo."

# cd to the script directory
cd /workspace/core

# Run the script
screen -dmS adam python training.py 1e-4 'adam' 'models/adam'
screen -dmS rms python training.py 1e-4 'rms' 'models/rms'

echo "Startup script executed successfully."
tail -f /dev/null