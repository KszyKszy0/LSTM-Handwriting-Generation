while true; do
  python3 core/training.py
  echo "App crashed. Restarting in 5 seconds..."
  sleep 5
done
