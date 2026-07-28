#!/bin/zsh
# Lanzador de la app local del drone HMS (doble click para iniciar).
cd "/Users/hms/Documents/Claude/Projects/determinacion de pesos y conteo bovinos utilizando drone"
source .venv/bin/activate
echo ""
echo "  Iniciando HMS Drone... se abre sola en el navegador."
echo "  Para cerrarla: cerra esta ventana de Terminal."
echo ""
exec streamlit run drone_app.py
