#!/usr/bin/env python3
"""Backup de la base de la app de gestion, con verificacion.

Por que existe: la base vive en Supabase plan Free, que NO hace backups
automaticos (verificado el 30/07/2026: el panel decia "No backups"). Un
borrado accidental o un problema de cuenta se llevaba todo: clientes,
lotes, pesadas, dietas e historial.

Que hace:
  1. pg_dump en formato custom (-Fc, ya comprimido) de la base entera.
  2. VERIFICA el dump antes de darlo por bueno: lo lista con
     pg_restore -l y comprueba que aparezcan las tablas criticas. Un
     backup que no se puede restaurar no es un backup, y un dump
     truncado pesa bytes y no falla solo.
  3. Manda el archivo por email al admin, reusando la config SMTP que
     ya usan las alertas (no necesita credenciales nuevas).

Uso:
    python3 scripts/backup_db.py                 # dump + verificacion + mail
    python3 scripts/backup_db.py --dry-run       # todo menos el envio
    python3 scripts/backup_db.py --solo-verificar ARCHIVO

Variables de entorno:
    DATABASE_URL   requerida. La misma que usa la app.
    PG_DUMP        opcional, ruta al binario. Default "pg_dump".
                   Tiene que ser version >= la del servidor (hoy 17.6).
    PG_RESTORE     opcional, default "pg_restore".
    SMTP_*         las que ya usan las alertas.
    DRY_RUN        "true" equivale a --dry-run.

Exit code 0 solo si el dump se genero Y paso la verificacion.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# Tablas sin las que un restore no sirve de nada. Si el dump no las
# trae, algo salio mal aunque pg_dump haya devuelto 0.
TABLAS_CRITICAS = [
    "clientes", "lotes", "dietas", "pesadas", "entregas_producto",
]


def _log(msg: str) -> None:
    print(f"[backup] {msg}", flush=True)


def generar_dump(destino: Path, url: str) -> None:
    """pg_dump -Fc de toda la base. Formato custom porque comprime solo
    y permite restaurar tablas sueltas con pg_restore."""
    pg_dump = os.getenv("PG_DUMP", "pg_dump")
    cmd = [
        pg_dump, url,
        "--format=custom",
        "--no-owner",          # el rol de Supabase no existe al restaurar
        "--no-privileges",     # los GRANT los rehace Supabase
        "--file", str(destino),
    ]
    _log(f"corriendo {pg_dump} ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"pg_dump fallo (codigo {r.returncode}): "
            f"{(r.stderr or '').strip()[:500]}"
        )


def verificar_dump(archivo: Path) -> dict:
    """Comprueba que el dump sea legible y traiga las tablas criticas.

    No alcanza con que el archivo exista: un dump cortado a la mitad
    tambien existe. pg_restore -l lee el indice interno, asi que si el
    archivo esta corrupto o truncado, falla aca y no dentro de seis
    meses cuando lo necesites.
    """
    if not archivo.exists():
        raise RuntimeError("el dump no existe")
    tam = archivo.stat().st_size
    if tam < 1024:
        raise RuntimeError(f"el dump pesa {tam} bytes: sospechosamente chico")

    pg_restore = os.getenv("PG_RESTORE", "pg_restore")
    r = subprocess.run(
        [pg_restore, "--list", str(archivo)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"pg_restore no pudo leer el dump: "
            f"{(r.stderr or '').strip()[:300]}"
        )

    listado = r.stdout
    tablas = {
        ln.rsplit(" ", 2)[-2]
        for ln in listado.splitlines()
        if " TABLE DATA " in ln
    }
    faltan = [t for t in TABLAS_CRITICAS if t not in tablas]
    if faltan:
        raise RuntimeError(
            f"el dump no trae datos de estas tablas: {', '.join(faltan)}"
        )

    return {
        "bytes": tam,
        "mb": round(tam / 1024 / 1024, 2),
        "n_tablas_con_datos": len(tablas),
        "tablas": sorted(tablas),
    }


def cuerpo_mail(info: dict, archivo: Path) -> tuple[str, str]:
    hoy = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M")
    filas = "".join(
        f"<tr><td style='padding:2px 10px 2px 0'>{t}</td></tr>"
        for t in info["tablas"]
    )
    html = f"""<html><body style="font-family:sans-serif;font-size:14px">
<h3 style="color:#1B3E27;margin-bottom:4px">Backup de la base — HMS</h3>
<p style="color:#444">Generado el {hoy}. Adjunto: <code>{archivo.name}</code></p>
<table style="border-collapse:collapse;font-size:13px">
  <tr><td style="padding-right:14px"><b>Tamano</b></td><td>{info['mb']} MB</td></tr>
  <tr><td><b>Tablas con datos</b></td><td>{info['n_tablas_con_datos']}</td></tr>
  <tr><td><b>Verificacion</b></td><td style="color:#0F6E56"><b>OK</b> — pg_restore pudo leerlo</td></tr>
</table>
<p style="color:#666;font-size:12px;margin-top:14px">Para restaurar:<br>
<code>pg_restore --no-owner --no-privileges -d "$DATABASE_URL" {archivo.name}</code></p>
<details><summary style="cursor:pointer;color:#666;font-size:12px">Tablas incluidas</summary>
<table>{filas}</table></details>
</body></html>"""
    texto = (
        f"Backup de la base HMS — {hoy}\n"
        f"Archivo: {archivo.name} ({info['mb']} MB)\n"
        f"Tablas con datos: {info['n_tablas_con_datos']}\n"
        f"Verificacion: OK (pg_restore pudo leer el dump)\n\n"
        f"Restaurar con:\n"
        f"  pg_restore --no-owner --no-privileges -d \"$DATABASE_URL\" {archivo.name}\n"
    )
    return html, texto


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Genera y verifica, pero no manda el mail.")
    ap.add_argument("--solo-verificar", metavar="ARCHIVO",
                    help="No genera nada: verifica un dump existente.")
    ap.add_argument("--salida", default=None,
                    help="Ruta del dump. Default: backups/ con la fecha.")
    args = ap.parse_args()

    dry = args.dry_run or (
        os.getenv("DRY_RUN", "").strip().lower() in ("1", "true", "yes", "si")
    )

    if args.solo_verificar:
        try:
            info = verificar_dump(Path(args.solo_verificar))
        except RuntimeError as e:
            _log(f"VERIFICACION FALLIDA: {e}")
            return 1
        _log(f"verificacion OK: {info['mb']} MB, "
             f"{info['n_tablas_con_datos']} tablas con datos")
        return 0

    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        _log("falta DATABASE_URL")
        return 1

    fecha = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    destino = Path(args.salida) if args.salida else (
        RAIZ / "backups" / f"hms-nutricion-{fecha}.dump"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)

    try:
        generar_dump(destino, url)
        info = verificar_dump(destino)
    except RuntimeError as e:
        _log(f"ERROR: {e}")
        return 1

    _log(f"dump OK: {destino.name} — {info['mb']} MB, "
         f"{info['n_tablas_con_datos']} tablas con datos")

    if dry:
        _log("DRY_RUN: no se manda el mail")
        return 0

    from src import alertas_email as ae
    cfg = ae.cargar_config_smtp()
    if not cfg:
        _log("sin config SMTP: el dump quedo generado pero no se envio")
        return 1
    destinatario = (cfg.get("admin_email") or cfg.get("from_email") or "").strip()
    if not destinatario:
        _log("sin admin_email ni from_email: no se a quien mandarlo")
        return 1

    html, texto = cuerpo_mail(info, destino)
    ok, msg = ae.enviar_email(
        cfg, [destinatario],
        f"Backup base HMS {fecha} — {info['mb']} MB",
        html, texto,
        embed_logo=False,
        con_bcc_admin=False,      # ya va al admin
        attachments=[str(destino)],
    )
    if not ok:
        _log(f"el mail no salio: {msg}")
        return 1
    _log(f"backup enviado a {destinatario}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
