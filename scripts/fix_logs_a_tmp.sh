#!/bin/bash
# Mueve los StandardOutPath/StandardErrorPath de los 8 plists HMS
# desde ~/Documents/Claude/Projects/.../data/logs/ a /tmp/.
#
# Por qué: macOS TCC bloquea a launchd para escribir en ~/Documents/,
# lo que causa exit EX_CONFIG (78) ANTES de que Python arranque.
# Moviendo los logs a /tmp/ (sin protección TCC), launchd puede crear
# los archivos y arrancar Python normalmente.
#
# Uso:  bash scripts/fix_logs_a_tmp.sh

set -u

AGENTES="$HOME/Library/LaunchAgents"
PROYECTO="$(cd "$(dirname "$0")/.." && pwd)"

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

echo "════════════════════════════════════════════════════════════"
echo "🔧 MIGRANDO LOGS DE LAUNCHD A /tmp"
echo "   (workaround para EX_CONFIG por TCC en ~/Documents/)"
echo "════════════════════════════════════════════════════════════"
echo ""

OK=0
FAIL=0

for plist in "${PLISTS[@]}"; do
    ARCH="$AGENTES/$plist.plist"
    echo "──────────────────────────────────────"
    echo "🔹 $plist"

    if [[ ! -f "$ARCH" ]]; then
        echo "   ⚠️  No instalado en $AGENTES — skip"
        continue
    fi

    # Backup
    cp "$ARCH" "$ARCH.bak.$(date +%Y%m%d_%H%M%S)"
    echo "   📋 Backup creado"

    # Reemplazar paths: cualquier *.log dentro de data/logs/ → /tmp/<label>-<...>.log
    /usr/libexec/PlistBuddy -c "Set :StandardOutPath /tmp/$plist-stdout.log" "$ARCH" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Add :StandardOutPath string /tmp/$plist-stdout.log" "$ARCH"
    /usr/libexec/PlistBuddy -c "Set :StandardErrorPath /tmp/$plist-stderr.log" "$ARCH" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Add :StandardErrorPath string /tmp/$plist-stderr.log" "$ARCH"

    echo "   ✏️  StandardOutPath  → /tmp/$plist-stdout.log"
    echo "   ✏️  StandardErrorPath → /tmp/$plist-stderr.log"

    # Recargar
    launchctl unload "$ARCH" 2>/dev/null
    sleep 1
    if launchctl load "$ARCH" 2>/dev/null; then
        echo "   ✅ Recargado en launchctl"
        OK=$((OK + 1))
    else
        echo "   ❌ Falló al recargar"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 RESUMEN"
echo "   ✅ OK: $OK"
echo "   ❌ Fallaron: $FAIL"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⏳ Esperando 10 segundos para que launchd estabilice..."
sleep 10
echo ""
echo "📋 Estado actual de los crons:"
launchctl list | grep com.hms

echo ""
echo "🔬 Forzando kickstart de alertas-diarias para probar..."
launchctl kickstart -k gui/$UID/com.hms.alertas-diarias
echo "   Esperando 60 segundos..."
sleep 60

echo ""
echo "📄 Resultado en /tmp/com.hms.alertas-diarias-stderr.log:"
echo "   ────────────────────────────"
if [[ -f /tmp/com.hms.alertas-diarias-stderr.log ]]; then
    cat /tmp/com.hms.alertas-diarias-stderr.log | tail -20
else
    echo "   (archivo no creado todavía)"
fi
echo "   ────────────────────────────"
echo ""
echo "📄 Últimas líneas del stdout en /tmp/com.hms.alertas-diarias-stdout.log:"
echo "   ────────────────────────────"
if [[ -f /tmp/com.hms.alertas-diarias-stdout.log ]]; then
    tail -15 /tmp/com.hms.alertas-diarias-stdout.log
else
    echo "   (archivo no creado todavía)"
fi
echo "   ────────────────────────────"
echo ""
echo "🏷️  Exit code final:"
launchctl list | grep com.hms.alertas-diarias
echo ""
echo "✅ FIN — Si el exit code pasó a 0, el problema era TCC en ~/Documents/."
echo "   Si sigue en 78, es algo más profundo."
