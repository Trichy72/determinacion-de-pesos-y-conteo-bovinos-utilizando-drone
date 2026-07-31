#!/usr/bin/env python3
"""Corrige la dirección de la app en el cron que la mantiene despierta.

El cron `keep_alive.yml` le pega cada 5 minutos a la app para que
Streamlit no la duerma. Hasta el 31/07/2026 apuntaba al subdominio
viejo, el de antes de renombrar el repo, así que hacía semanas que le
tocaba el timbre a una dirección muerta. Y como ese paso termina
siempre en éxito pase lo que pase, nadie se enteró: la app se dormía
igual y tardaba en abrir en el campo.

Este script hace dos cosas:
  1. Cambia la dirección vieja por la nueva.
  2. Agrega un aviso visible en GitHub cuando el ping no contesta 200,
     sin romper el workflow — para que la próxima vez se vea.

Es seguro correrlo dos veces: si ya está arreglado, avisa y no toca
nada.

Uso:   python3 scripts/arreglar_ping.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO = RAIZ / ".github" / "workflows" / "keep_alive.yml"

URL_VIEJA = (
    "https://determinacion-de-pesos-y-conteo-bovinos-utilizando-drone"
    "-9s6v3.streamlit.app/"
)
URL_NUEVA = "https://hms-gestion-ganadera.streamlit.app/"

BLOQUE_VIEJO = '''          CODE=$(curl -sS -o /dev/null -w "%{http_code}" -L --max-redirs 100 --max-time 90 "$URL" || echo "err")
          echo "Ping HTTP $CODE - $(date -u)"
          exit 0'''

BLOQUE_NUEVO = '''          CODE=$(curl -sS -o /dev/null -w "%{http_code}" -L --max-redirs 100 --max-time 90 "$URL" || echo "err")
          echo "Ping HTTP $CODE - $(date -u)"
          if [ "$CODE" != "200" ]; then
            echo "::warning title=Ping sin respuesta::La app respondio '$CODE' en $URL. Si se repite, revisar que la URL siga siendo la correcta."
          fi
          exit 0'''

print("=" * 66)
print("ARREGLO DEL CRON QUE MANTIENE LA APP DESPIERTA")
print("=" * 66)

if not ARCHIVO.exists():
    sys.exit(f"\nNo encontré el archivo:\n  {ARCHIVO}\n"
             "¿Estás parado en la carpeta del proyecto?")

texto = ARCHIVO.read_text(encoding="utf-8")
original = texto
cambios = []

if URL_VIEJA in texto:
    texto = texto.replace(URL_VIEJA, URL_NUEVA)
    cambios.append("dirección de la app corregida")
elif URL_NUEVA in texto:
    print("\n· La dirección ya estaba bien.")
else:
    print("\n! No encontré ninguna de las dos direcciones conocidas.")
    print("  Alguien ya lo editó a mano. No toco nada por las dudas.")
    sys.exit(1)

if BLOQUE_VIEJO in texto:
    texto = texto.replace(BLOQUE_VIEJO, BLOQUE_NUEVO)
    cambios.append("aviso agregado para cuando el ping no conteste")
elif "::warning title=Ping sin respuesta" in texto:
    print("· El aviso ya estaba puesto.")

if not cambios:
    print("\nYa estaba todo arreglado. No hice nada.")
    sys.exit(0)

# Copia de seguridad antes de escribir, por si acaso.
respaldo = ARCHIVO.with_suffix(".yml.antes-del-arreglo")
respaldo.write_text(original, encoding="utf-8")
ARCHIVO.write_text(texto, encoding="utf-8")

print("\nHecho:")
for c in cambios:
    print(f"  · {c}")
print(f"\nGuardé una copia del archivo original en:\n  {respaldo.name}")
print("\nAhora falta subirlo: commit y Push origin en GitHub Desktop.")
print("El cron corre cada 5 minutos, así que se aplica solo enseguida.")
