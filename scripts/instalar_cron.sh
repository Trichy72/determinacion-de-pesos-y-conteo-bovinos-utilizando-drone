#!/bin/bash
# Instala los jobs de alertas en macOS (launchd) o Linux (cron).
#  - Diario 08:00: email digest + WhatsApp resumen al admin
#  - Cada 1 hora: WhatsApp instantáneo si hay alertas críticas nuevas
#
# Uso:
#   ./scripts/instalar_cron.sh

set -e

PROYECTO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> Proyecto: $PROYECTO_DIR"

OS="$(uname -s)"

if [ "$OS" = "Darwin" ]; then
    echo "==> macOS detectado, usando launchd"
    mkdir -p "$HOME/Library/LaunchAgents"
    mkdir -p "$PROYECTO_DIR/data/logs"

    # ---- Job 1: Diario 08:00 ----
    PLIST_DIA_T="$PROYECTO_DIR/scripts/com.hms.alertas-diarias.plist"
    PLIST_DIA_D="$HOME/Library/LaunchAgents/com.hms.alertas-diarias.plist"
    sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROYECTO_DIR|g" "$PLIST_DIA_T" > "$PLIST_DIA_D"
    launchctl unload "$PLIST_DIA_D" 2>/dev/null || true
    launchctl load "$PLIST_DIA_D"
    echo "  ✓ Job DIARIO instalado (08:00 AM): $PLIST_DIA_D"

    # ---- Job 1b: Rescate 08:30 (reintenta si Open-Meteo falló a las 8) ----
    PLIST_RES_T="$PROYECTO_DIR/scripts/com.hms.alertas-rescate.plist"
    PLIST_RES_D="$HOME/Library/LaunchAgents/com.hms.alertas-rescate.plist"
    sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROYECTO_DIR|g" "$PLIST_RES_T" > "$PLIST_RES_D"
    launchctl unload "$PLIST_RES_D" 2>/dev/null || true
    launchctl load "$PLIST_RES_D"
    echo "  ✓ Job RESCATE instalado (08:30 AM): $PLIST_RES_D"

    # ---- Job 1c: Tarde 18:00 (pronóstico nocturno, solo si hay alertas) ----
    PLIST_TAR_T="$PROYECTO_DIR/scripts/com.hms.alertas-tarde.plist"
    PLIST_TAR_D="$HOME/Library/LaunchAgents/com.hms.alertas-tarde.plist"
    sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROYECTO_DIR|g" "$PLIST_TAR_T" > "$PLIST_TAR_D"
    launchctl unload "$PLIST_TAR_D" 2>/dev/null || true
    launchctl load "$PLIST_TAR_D"
    echo "  ✓ Job TARDE instalado (18:00 PM): $PLIST_TAR_D"

    # ---- Job 1d: Semanal lunes 7:30 (pronóstico de la semana) ----
    PLIST_SEM_T="$PROYECTO_DIR/scripts/com.hms.alertas-semanales.plist"
    PLIST_SEM_D="$HOME/Library/LaunchAgents/com.hms.alertas-semanales.plist"
    sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROYECTO_DIR|g" "$PLIST_SEM_T" > "$PLIST_SEM_D"
    launchctl unload "$PLIST_SEM_D" 2>/dev/null || true
    launchctl load "$PLIST_SEM_D"
    echo "  ✓ Job SEMANAL instalado (lunes 07:30): $PLIST_SEM_D"

    # ---- Job 1e: Update miércoles 7:30 (refresh del pronóstico) ----
    PLIST_UPD_T="$PROYECTO_DIR/scripts/com.hms.alertas-semanales-update.plist"
    PLIST_UPD_D="$HOME/Library/LaunchAgents/com.hms.alertas-semanales-update.plist"
    sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROYECTO_DIR|g" "$PLIST_UPD_T" > "$PLIST_UPD_D"
    launchctl unload "$PLIST_UPD_D" 2>/dev/null || true
    launchctl load "$PLIST_UPD_D"
    echo "  ✓ Job UPDATE instalado (miércoles 07:30): $PLIST_UPD_D"

    # ---- Job 1f: Informe demanda lunes 8:00 (al admin, por cliente) ----
    PLIST_DEM_T="$PROYECTO_DIR/scripts/com.hms.informe-demanda-semanal.plist"
    PLIST_DEM_D="$HOME/Library/LaunchAgents/com.hms.informe-demanda-semanal.plist"
    sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROYECTO_DIR|g" "$PLIST_DEM_T" > "$PLIST_DEM_D"
    launchctl unload "$PLIST_DEM_D" 2>/dev/null || true
    launchctl load "$PLIST_DEM_D"
    echo "  ✓ Job INFORME DEMANDA instalado (lunes 08:00): $PLIST_DEM_D"

    # ---- Job 2: Cada hora ----
    PLIST_CR_T="$PROYECTO_DIR/scripts/com.hms.alertas-criticas.plist"
    PLIST_CR_D="$HOME/Library/LaunchAgents/com.hms.alertas-criticas.plist"
    sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROYECTO_DIR|g" "$PLIST_CR_T" > "$PLIST_CR_D"
    launchctl unload "$PLIST_CR_D" 2>/dev/null || true
    launchctl load "$PLIST_CR_D"
    echo "  ✓ Job HORARIO instalado (cada 60 min): $PLIST_CR_D"

    echo ""
    echo "Verificación:    launchctl list | grep hms"
    echo "Correr ahora:"
    echo "    launchctl start com.hms.alertas-diarias"
    echo "    launchctl start com.hms.alertas-rescate"
    echo "    launchctl start com.hms.alertas-tarde"
    echo "    launchctl start com.hms.alertas-semanales"
    echo "    launchctl start com.hms.alertas-semanales-update"
    echo "    launchctl start com.hms.informe-demanda-semanal"
    echo "    launchctl start com.hms.alertas-criticas"
    echo "Desinstalar:"
    echo "    launchctl unload $PLIST_DIA_D && rm $PLIST_DIA_D"
    echo "    launchctl unload $PLIST_RES_D && rm $PLIST_RES_D"
    echo "    launchctl unload $PLIST_TAR_D && rm $PLIST_TAR_D"
    echo "    launchctl unload $PLIST_SEM_D && rm $PLIST_SEM_D"
    echo "    launchctl unload $PLIST_UPD_D && rm $PLIST_UPD_D"
    echo "    launchctl unload $PLIST_DEM_D && rm $PLIST_DEM_D"
    echo "    launchctl unload $PLIST_CR_D && rm $PLIST_CR_D"

elif [ "$OS" = "Linux" ]; then
    echo "==> Linux detectado, usando cron"
    mkdir -p "$PROYECTO_DIR/data/logs"

    LINEA_DIA="0 8 * * * cd $PROYECTO_DIR && /usr/bin/python3 scripts/alertas_diarias.py >> $PROYECTO_DIR/data/logs/cron-diaria.log 2>&1"
    LINEA_RES="30 8 * * * cd $PROYECTO_DIR && /usr/bin/python3 scripts/alertas_diarias.py >> $PROYECTO_DIR/data/logs/cron-rescate.log 2>&1"
    LINEA_TAR="0 18 * * * cd $PROYECTO_DIR && /usr/bin/python3 scripts/alertas_tarde.py >> $PROYECTO_DIR/data/logs/cron-tarde.log 2>&1"
    LINEA_HOR="0 */1 * * * cd $PROYECTO_DIR && /usr/bin/python3 scripts/alertas_criticas.py >> $PROYECTO_DIR/data/logs/cron-criticas.log 2>&1"

    (crontab -l 2>/dev/null | grep -v "alertas_diarias.py" | grep -v "alertas_tarde.py" | grep -v "alertas_criticas.py" ; \
     echo "$LINEA_DIA" ; echo "$LINEA_RES" ; echo "$LINEA_TAR" ; echo "$LINEA_HOR") | crontab -

    echo "  ✓ Cron instalado:"
    echo "    DIARIO:   $LINEA_DIA"
    echo "    RESCATE:  $LINEA_RES"
    echo "    TARDE:    $LINEA_TAR"
    echo "    HORARIO:  $LINEA_HOR"
    echo ""
    echo "Ver: crontab -l    Editar: crontab -e"

else
    echo "==> SO no reconocido ($OS). Configurá manualmente."
    exit 1
fi

echo ""
echo "==> Probá ahora con:"
echo "    python3 scripts/alertas_diarias.py --dry-run"
echo "    python3 scripts/alertas_criticas.py --dry-run"
