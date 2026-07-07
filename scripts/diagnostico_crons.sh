#!/bin/bash
# Diagnóstico completo de todos los cron jobs HMS.
# Responde a "¿por qué no llegan las alertas?".
#
# Verifica para cada plist:
#   1. ¿Está instalado en ~/Library/LaunchAgents/?
#   2. ¿Está cargado en launchctl?
#   3. ¿Cuándo corrió por última vez?
#   4. ¿Hay errores en su log?
#
# Uso:  bash scripts/diagnostico_crons.sh

set -u

PROYECTO="$(cd "$(dirname "$0")/.." && pwd)"
AGENTES_DIR="$HOME/Library/LaunchAgents"

# Plists esperados del proyecto
PLISTS=(
    "com.hms.alertas-diarias"
    "com.hms.alertas-tarde"
    "com.hms.alertas-semanales"
    "com.hms.alertas-semanales-update"
    "com.hms.alertas-criticas"
    "com.hms.alertas-rescate"
    "com.hms.informe-demanda-semanal"
    "com.hms.pedido-carga"
)

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🩺 DIAGNÓSTICO DE CRON JOBS HMS"
echo "   Proyecto: $PROYECTO"
echo "   Agentes:  $AGENTES_DIR"
echo "   Fecha:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
echo ""

CARGADOS_OK=0
INSTALADOS_NO_CARGADOS=0
NO_INSTALADOS=0

for plist in "${PLISTS[@]}"; do
    echo "──────────────────────────────────────────────────────"
    echo "🔹 $plist"

    # 1. ¿Está en ~/Library/LaunchAgents/?
    INSTALADO="$AGENTES_DIR/$plist.plist"
    if [[ -f "$INSTALADO" ]]; then
        echo "   ✅ Instalado en $AGENTES_DIR"
    else
        echo "   ❌ NO instalado en $AGENTES_DIR"
        echo "      Para instalar:"
        echo "      cp \"$PROYECTO/scripts/$plist.plist\" \\"
        echo "         \"$AGENTES_DIR/\""
        echo "      launchctl load \"$AGENTES_DIR/$plist.plist\""
        NO_INSTALADOS=$((NO_INSTALADOS + 1))
        continue
    fi

    # 2. ¿Está cargado en launchctl?
    STATUS_LINE=$(launchctl list 2>/dev/null | grep -w "$plist" || true)
    if [[ -n "$STATUS_LINE" ]]; then
        # Columna 1 = PID (- si no está corriendo), 2 = status code, 3 = label
        PID=$(echo "$STATUS_LINE" | awk '{print $1}')
        EXIT_CODE=$(echo "$STATUS_LINE" | awk '{print $2}')
        echo "   ✅ Cargado en launchctl"
        if [[ "$EXIT_CODE" != "0" ]]; then
            echo "   ⚠️  Último exit code: $EXIT_CODE (≠ 0 indica error)"
        fi
        CARGADOS_OK=$((CARGADOS_OK + 1))
    else
        echo "   ❌ NO cargado en launchctl"
        echo "      Para cargarlo:"
        echo "      launchctl load \"$INSTALADO\""
        INSTALADOS_NO_CARGADOS=$((INSTALADOS_NO_CARGADOS + 1))
    fi

    # 3. Logs (buscar en ~/Library/Logs/HMS — ubicación actual —
    #    y en /tmp + data/logs por compatibilidad histórica)
    for LOG_DIR in "$HOME/Library/Logs/HMS" "/tmp" "$HOME/Library/Logs" "$PROYECTO/data/logs"; do
        for LOG_BASE in "$plist" "$plist-stdout" "$plist-stderr" \
                "${plist#com.hms.}" \
                "${plist#com.hms.}-stderr" "${plist#com.hms.}-stdout" \
                "launchd-${plist#com.hms.alertas-}-stdout" \
                "launchd-${plist#com.hms.alertas-}-stderr" \
                "launchd-${plist#com.hms.}-stdout" \
                "launchd-${plist#com.hms.}-stderr"; do
            for LOG_FILE in \
                "$LOG_DIR/$LOG_BASE.log" \
                "$LOG_DIR/$LOG_BASE.err" \
                "$LOG_DIR/$LOG_BASE.out"; do
                if [[ -f "$LOG_FILE" ]]; then
                    MOD=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" \
                          "$LOG_FILE" 2>/dev/null)
                    SIZE=$(stat -f "%z" "$LOG_FILE" 2>/dev/null)
                    echo "   📄 Log: $LOG_FILE ($SIZE bytes, mod $MOD)"
                fi
            done
        done
    done

    # 4. StandardErrorPath / StandardOutPath del plist
    if [[ -f "$INSTALADO" ]]; then
        STDERR_LOG=$(/usr/libexec/PlistBuddy -c "Print :StandardErrorPath" \
                     "$INSTALADO" 2>/dev/null || true)
        STDOUT_LOG=$(/usr/libexec/PlistBuddy -c "Print :StandardOutPath" \
                     "$INSTALADO" 2>/dev/null || true)
        if [[ -n "$STDERR_LOG" && -f "$STDERR_LOG" ]]; then
            MOD=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" \
                  "$STDERR_LOG" 2>/dev/null)
            SIZE=$(stat -f "%z" "$STDERR_LOG" 2>/dev/null)
            echo "   📄 stderr: $STDERR_LOG ($SIZE bytes, mod $MOD)"
            if [[ "$SIZE" -gt 0 ]]; then
                echo "      Últimas 5 líneas del stderr:"
                tail -5 "$STDERR_LOG" 2>/dev/null | sed 's/^/        /'
            fi
        fi
        if [[ -n "$STDOUT_LOG" && -f "$STDOUT_LOG" ]]; then
            MOD=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" \
                  "$STDOUT_LOG" 2>/dev/null)
            SIZE=$(stat -f "%z" "$STDOUT_LOG" 2>/dev/null)
            echo "   📄 stdout: $STDOUT_LOG ($SIZE bytes, mod $MOD)"
        fi
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════════"
echo "📊 RESUMEN"
echo "   ✅ Cargados y funcionando: $CARGADOS_OK"
echo "   ⚠️  Instalados pero NO cargados: $INSTALADOS_NO_CARGADOS"
echo "   ❌ NO instalados: $NO_INSTALADOS"
echo "════════════════════════════════════════════════════════════"
echo ""

if [[ $NO_INSTALADOS -gt 0 ]]; then
    echo "🔧 ACCIÓN: Hay plists NO instalados — copialos a "
    echo "   $AGENTES_DIR/ y cargá con launchctl load."
    echo ""
fi
if [[ $INSTALADOS_NO_CARGADOS -gt 0 ]]; then
    echo "🔧 ACCIÓN: Hay plists instalados pero NO cargados."
    echo "   Para recargarlos TODOS de una sola vez:"
    echo ""
    for plist in "${PLISTS[@]}"; do
        if [[ -f "$AGENTES_DIR/$plist.plist" ]]; then
            STATUS=$(launchctl list 2>/dev/null | grep -w "$plist" || true)
            if [[ -z "$STATUS" ]]; then
                echo "   launchctl load \"$AGENTES_DIR/$plist.plist\""
            fi
        fi
    done
    echo ""
fi
if [[ $CARGADOS_OK -eq ${#PLISTS[@]} ]]; then
    echo "✅ Todos los crons están cargados."
    echo "   Si igual no llegan emails, revisar:"
    echo "   - Configuración SMTP (data/smtp_config.json)"
    echo "   - Logs específicos arriba (stderr de cada cron)"
    echo "   - launchctl print gui/\$UID/com.hms.alertas-diarias"
fi
echo ""
