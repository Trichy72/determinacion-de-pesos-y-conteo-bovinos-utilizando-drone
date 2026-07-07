"""File lock global para scripts de cron.

Si dos procesos del mismo cron arrancan simultáneamente (por ejemplo,
launchd despierta la Mac y dispara un job atrasado mientras corre el
StartCalendarInterval normal), ambos pasarían el chequeo de dedup en DB
porque arrancan antes de que el primero pueda registrar el envío. Esto
genera emails y WhatsApp duplicados.

El lock se toma en `main()` con `adquirir_lock_proceso(nombre)`. Si otro
proceso ya lo tiene, retorna `None` y el script debe salir 0
silenciosamente.

Uso:
    from src.locking import adquirir_lock_proceso

    def main():
        lock_fd = adquirir_lock_proceso("alertas_diarias")
        if lock_fd is None:
            print("Otro proceso ya está corriendo. Saliendo.")
            return 0
        try:
            # ... lógica del script ...
        finally:
            liberar_lock(lock_fd)
"""
from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
from typing import Optional


def _lock_path(nombre: str) -> Path:
    """Ruta del archivo de lock para `nombre`. Se guarda en /tmp para
    que sobreviva al SIGKILL del proceso (no necesita cleanup explícito)
    y para no contaminar el repo."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in nombre)
    return Path(tempfile.gettempdir()) / f"hms_lock_{safe}.lock"


def adquirir_lock_proceso(nombre: str) -> Optional[int]:
    """Intenta tomar un lock exclusivo no bloqueante.

    Retorna el file descriptor abierto si lo tomó.
    Retorna None si otro proceso ya lo tiene (señal de que hay otra
    instancia corriendo y este proceso debe abortar silenciosamente).
    """
    path = _lock_path(nombre)
    try:
        # Abrir en modo append para no truncar y poder escribir el PID.
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        # Otro proceso lo tiene.
        os.close(fd)
        return None

    # Escribir PID como referencia (no se usa para nada lógico, solo debug).
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        pass

    return fd


def liberar_lock(fd: Optional[int]) -> None:
    """Libera el lock y cierra el descriptor. Idempotente."""
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
