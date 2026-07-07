#!/bin/bash
# Migra los logs de launchd desde /tmp (temporal, se borra al reboot) a
# ~/Library/Logs/HMS/ (persistente, sin protección TCC).
#
# Contexto: macOS TCC bloquea a launchd para escribir en ~/Documents/, lo
# que genera exit EX_CONFIG (78) antes de que Python pueda arrancar. La
# fix anterior (fix_logs_a_tmp.sh) movió los logs a /tmp/ — funciona, pero
# /tmp se limpia al reboot, perdiendo historia.
#
# Este script mueve los logs a ~/Library/Logs/HMS/ que NO está protegida
# por TCC y persiste entre reboots. También copia el contenido actual
# de los logs en /tmp/ a la nueva ubicación (preserva history).
#
# Uso:  bash scripts/migrar_logs_a_library.sh

set -u

AGENTES="$HOME/Library/LaunchAgents"
DESTINO="$HOME/Library/Logs/HMS"

# Mapping: <label-corto> en plist → <log-name>
declare -a PLISTS=(
    "com.hms.alertas-diarias:diarias"
    "com.hms.alertas-tarde:tarde"
    "com.hms.alertas-semanales:semanal"
    "com.hms.alertas-semanales-update:semanal-update"
    "com.hms.alertas-criticas:criticas"
    "com.hms.alertas-rescate:rescate"
    "com.hms.informe-demanda-semanal:demanda"
    "com.hms.pedido-carga:pedido-carga"
)

echo "════════════════════════════════════════════════════════════"
echo "📦 MIGRANDO LOGS DE LAUNCHD"
echo "   /tmp/  →  ~/Library/Logs/HMS/"
echo "   (persistente entre reboots, sin protección TCC)"
echo "════════════════════════════════════════════════════════════"
echo ""

# 1. Crear destino
mkdir -p "$DESTINO"
echo "📁 Destino creado: $DESTINO"
echo ""

OK=0
FAIL=0

for entry in "${PLISTS[@]}"; do
    PLIST="${entry%%:*}"
    LOGNAME="${entry##*:}"
    ARCH="$AGENTES/$PLIST.plist"

    echo "──────────────────────────────────────"
    echo "🔹 $PLIST"

    if [[ ! -f "$ARCH" ]]; then
        echo "   ⚠️  No instalado — skip"
        continue
    fi

    # Backup del plist
    cp "$ARCH" "$ARCH.bak.migrar_library.$(date +%Y%m%d_%H%M%S)"

    # Preservar logs viejos de /tmp/ (si existen)
    OLD_STDOUT="/tmp/$PLIST-stdout.log"
    OLD_STDERR="/tmp/$PLIST-stderr.log"
    NEW_STDOUT="$DESTINO/launchd-$LOGNAME-stdout.log"
    NEW_STDERR="$DESTINO/launchd-$LOGNAME-stderr.log"

    if [[ -f "$OLD_STDOUT" ]] && [[ -s "$OLD_STDOUT" ]]; then
        cat "$OLD_STDOUT" >> "$NEW_STDOUT"
        echo "   📋 stdout preservado de /tmp"
    fi
    if [[ -f "$OLD_STDERR" ]] && [[ -s "$OLD_STDERR" ]]; then
        cat "$OLD_STDERR" >> "$NEW_STDERR"
        echo "   📋 stderr preservado de /tmp"
    fi

    # Actualizar paths en el plist
    /usr/libexec/PlistBuddy -c "Set :StandardOutPath $NEW_STDOUT" "$ARCH" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Add :StandardOutPath string $NEW_STDOUT" "$ARCH"
    /usr/libexec/PlistBuddy -c "Set :StandardErrorPath $NEW_STDERR" "$ARCH" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Add :StandardErrorPath string $NEW_STDERR" "$ARCH"

    echo "   ✏️  stdout → $NEW_STDOUT"
    echo "   ✏️  stderr → $NEW_STDERR"

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
echo "   ✅ Migrados OK: $OK"
echo "   ❌ Fallaron: $FAIL"
echo "════════════════════════════════════════════════════════════"
echo ""

if [[ $OK -eq 8 ]] && [[ $FAIL -eq 0 ]]; then
    echo "✅ Todos los plists migrados correctamente."
    echo ""
    echo "📍 Los nuevos logs viven en: $DESTINO/"
    echo "   - launchd-diarias-stdout.log + launchd-diarias-stderr.log"
    echo "   - launchd-tarde-stdout.log   + launchd-tarde-stderr.log"
    echo "   - launchd-semanal-...        + launchd-semanal-...-stderr.log"
    echo "   - etc."
    echo ""
    echo "🧹 (Opcional) Los logs viejos en /tmp/ se borrarán solos al"
    echo "   reboot. Si querés liberar espacio ahora:"
    echo "   rm /tmp/com.hms.*-{stdout,stderr}.log"
    echo ""
    echo "⏳ Esperando 10 segundos para verificar exit codes..."
    sleep 10
    echo ""
    echo "📋 Estado actual:"
    launchctl list | grep com.hms
else
    echo "⚠️  Algunos plists fallaron. Revisá los mensajes arriba."
fi
echo ""
