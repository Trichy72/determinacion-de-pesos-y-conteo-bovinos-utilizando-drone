"""Script de prueba: manda al admin (no al cliente) los emails de las
3 alertas del paquete logístico:

  1. Stock HMS bajo (1 producto crítico + 1 moderado consolidado)
  2. Fin de carga del silocomedero (próximo a agotarse mañana)
  3. Cambio de fase en plan de adaptación (mañana arranca fase nueva)

Es solo preview visual — usa datos ficticios realistas (Bergondi /
Pezzola) y manda los 3 mails con prefijo [PRUEBA] en el subject.
No escribe en la DB, no notifica al cliente.

Uso:
    .venv/bin/python scripts/test_alertas_nuevas.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import alertas_email as ae


def main() -> int:
    cfg = ae.cargar_config_smtp() or {}
    if not cfg.get("host"):
        print("❌ No hay config SMTP cargada. "
              "Andá a Configuración primero.")
        return 1

    destinatario_prueba = cfg.get("admin_email") or cfg.get("from_email")
    print(f"📨 Destinatario de prueba: {destinatario_prueba}")
    print("    (es vos, no el cliente — esto es solo para ver el formato)")

    # ════════════════════════════════════════════════════════════════
    # ALERTA 1 — STOCK HMS BAJO
    # ════════════════════════════════════════════════════════════════
    cliente_stock = {
        "nombre": "Miguel Bergondi",
        "establecimiento": "La Cancha",
    }
    contacto_stock = {
        "nombre": "Miguel Bergondi",
        "email": destinatario_prueba,
    }
    productos_stock = [
        {
            "lote_ident": "Engorde vacas",
            "producto": "Fibrogreen plus",
            "kg_restantes": 90,
            "consumo_kg_dia": 60.0,
            "dias_restantes": 8,
            "fecha_agotamiento": "2026-05-29",
        },
        {
            "lote_ident": "Recría B",
            "producto": "Fibroter",
            "kg_restantes": 168,
            "consumo_kg_dia": 12.0,
            "dias_restantes": 14,
            "fecha_agotamiento": "2026-06-04",
        },
    ]

    # ════════════════════════════════════════════════════════════════
    # ALERTA 2 — FIN DE CARGA SILOCOMEDERO
    # ════════════════════════════════════════════════════════════════
    cliente_silo = {
        "nombre": "Miguel Bergondi",
        "establecimiento": "La Cancha",
    }
    contacto_silo = {
        "nombre": "Miguel Bergondi",
        "email": destinatario_prueba,
    }
    lotes_silo = [
        {
            "lote_id": 1,
            "lote_ident": "Engorde vacas",
            "categoria": "vaca adulta",
            "kg_cargados": 360.0,
            "fecha_carga": "2026-05-18",
            "consumo_diario_kg": 90.0,
            "kg_consumidos_acumulados": 270.0,
            "kg_restantes": 90.0,
            "dias_restantes": 1,
            "fecha_agotamiento": "2026-05-22",
        },
    ]

    # ════════════════════════════════════════════════════════════════
    # ALERTA 3 — CAMBIO DE FASE EN PLAN DE ADAPTACIÓN
    # ════════════════════════════════════════════════════════════════
    cliente_fase = {
        "nombre": "Miguel Bergondi",
        "establecimiento": "La Cancha",
    }
    contacto_fase = {
        "nombre": "Miguel Bergondi",
        "email": destinatario_prueba,
    }
    # Cambio realista de Bergondi: Fase 3 → Fase 4 con dieta completa
    # (Fibrogreen + Maíz + Rollo). El rollo va a libre disposición —
    # el animal regula su consumo.
    cambios_fase = [
        {
            "lote_id": 1,
            "lote_ident": "Engorde vacas",
            "categoria": "vaca adulta",
            "cantidad_animales": 30,
            "fecha_cambio": "2026-05-22",
            "dias_para_cambio": 1,
            "fase_actual": {
                "fecha": "2026-05-15",
                "observaciones": "Fase 3",
                "composicion": [
                    {"nombre": "Fibrogreen plus", "pct_ms": 32.0,
                     "kg_tal_cual": 4.7},
                    {"nombre": "Maíz molido", "pct_ms": 28.0,
                     "kg_tal_cual": 4.0},
                    {"nombre": "Rollo alfalfa", "pct_ms": 40.0,
                     "kg_tal_cual": 5.5},
                ],
                "costo_dia": 3242,
            },
            "fase_nueva": {
                "fecha": "2026-05-22",
                "observaciones": "Fase 4",
                "composicion": [
                    {"nombre": "Fibrogreen plus", "pct_ms": 38.0,
                     "kg_tal_cual": 5.68},
                    {"nombre": "Maíz molido", "pct_ms": 32.0,
                     "kg_tal_cual": 5.0},
                    {"nombre": "Rollo alfalfa", "pct_ms": 30.0,
                     "kg_tal_cual": 4.5},
                ],
                "costo_dia": 3864,
                # Última fase del plan: vigente hasta nuevo aviso.
                # Si quisieras simular una fase intermedia, cargá
                # fecha_fin: "2026-05-28" y duracion_dias: 7.
                "fecha_fin": None,
                "duracion_dias": None,
            },
            "diff": [
                {
                    "ingrediente": "Fibrogreen plus",
                    "kg_actual": 4.7, "kg_nueva": 5.68,
                    "delta_kg": 0.98,
                    "pct_actual": 32.0, "pct_nueva": 38.0,
                },
                {
                    "ingrediente": "Maíz molido",
                    "kg_actual": 4.0, "kg_nueva": 5.0,
                    "delta_kg": 1.0,
                    "pct_actual": 28.0, "pct_nueva": 32.0,
                },
                {
                    "ingrediente": "Rollo alfalfa",
                    "kg_actual": 5.5, "kg_nueva": 4.5,
                    "delta_kg": -1.0,
                    "pct_actual": 40.0, "pct_nueva": 30.0,
                },
            ],
        },
    ]

    # ════════════════════════════════════════════════════════════════
    # INFORME 4 — DEMANDA CONSOLIDADA POR CLIENTE (interno, no cliente)
    # ════════════════════════════════════════════════════════════════
    # Mauricio recibe un resumen logístico con todos los lotes del
    # cliente y la demanda diaria/semanal/mensual de cada insumo,
    # marcando productos HMS. Datos simulados de Bergondi con 2 lotes:
    # Engorde (30 vacas) + Recría (50 terneros).
    cliente_demanda = {
        "nombre": "Miguel Bergondi",
        "establecimiento": "La Cancha",
    }
    demanda_simulada = {
        "cliente_id": 0,
        "fecha_referencia": "2026-05-18",
        "lotes": [
            {
                "lote_id": 1,
                "lote_ident": "Engorde vacas",
                "categoria": "vaca adulta",
                "cantidad_animales": 30,
                "fase_vigente": "Fase 4",
                "fecha_dieta": "2026-05-22",
                "ingredientes": [
                    {"nombre": "Fibrogreen plus",
                     "kg_animal_dia": 5.68, "kg_lote_dia": 170.4,
                     "kg_lote_semana": 1192.8,
                     "es_hms": True, "es_libre_disposicion": False},
                    {"nombre": "Maíz molido",
                     "kg_animal_dia": 5.0, "kg_lote_dia": 150.0,
                     "kg_lote_semana": 1050.0,
                     "es_hms": False, "es_libre_disposicion": False},
                    {"nombre": "Rollo alfalfa",
                     "kg_animal_dia": 4.5, "kg_lote_dia": 135.0,
                     "kg_lote_semana": 945.0,
                     "es_hms": False, "es_libre_disposicion": True},
                ],
                "mezcla_total_kg_dia": 320.4,
            },
            {
                "lote_id": 2,
                "lote_ident": "Recría B",
                "categoria": "ternero",
                "cantidad_animales": 50,
                "fase_vigente": "Recría inicial",
                "fecha_dieta": "2026-05-01",
                "ingredientes": [
                    {"nombre": "Fibroter",
                     "kg_animal_dia": 1.2, "kg_lote_dia": 60.0,
                     "kg_lote_semana": 420.0,
                     "es_hms": True, "es_libre_disposicion": False},
                    {"nombre": "Maíz molido",
                     "kg_animal_dia": 1.5, "kg_lote_dia": 75.0,
                     "kg_lote_semana": 525.0,
                     "es_hms": False, "es_libre_disposicion": False},
                    {"nombre": "Rollo alfalfa",
                     "kg_animal_dia": 2.0, "kg_lote_dia": 100.0,
                     "kg_lote_semana": 700.0,
                     "es_hms": False, "es_libre_disposicion": True},
                ],
                "mezcla_total_kg_dia": 135.0,
            },
        ],
        "total_cliente": {
            "ingredientes": [
                {"nombre": "Fibrogreen plus", "kg_dia": 170.4,
                 "kg_semana": 1192.8, "kg_mes": 5112.0,
                 "es_hms": True, "es_libre_disposicion": False,
                 "lotes_que_lo_usan": 1},
                {"nombre": "Fibroter", "kg_dia": 60.0,
                 "kg_semana": 420.0, "kg_mes": 1800.0,
                 "es_hms": True, "es_libre_disposicion": False,
                 "lotes_que_lo_usan": 1},
                {"nombre": "Maíz molido", "kg_dia": 225.0,
                 "kg_semana": 1575.0, "kg_mes": 6750.0,
                 "es_hms": False, "es_libre_disposicion": False,
                 "lotes_que_lo_usan": 2},
                {"nombre": "Rollo alfalfa", "kg_dia": 235.0,
                 "kg_semana": 1645.0, "kg_mes": 7050.0,
                 "es_hms": False, "es_libre_disposicion": True,
                 "lotes_que_lo_usan": 2},
            ],
            "mezcla_total_kg_dia": 455.4,
            "cantidad_animales_total": 80,
        },
    }

    pruebas = [
        (
            "ALERTA 1 — STOCK HMS BAJO (consolidado, 2 productos)",
            lambda: ae.componer_alerta_stock_cliente(
                cliente_stock, contacto_stock, productos_stock,
            ),
        ),
        (
            "ALERTA 2 — FIN DE CARGA SILOCOMEDERO (1 día)",
            lambda: ae.componer_alerta_silocomedero_cliente(
                cliente_silo, contacto_silo, lotes_silo,
            ),
        ),
        (
            "ALERTA 3 — CAMBIO DE FASE (Fase 3 → Fase 4)",
            lambda: ae.componer_alerta_cambio_fase_cliente(
                cliente_fase, contacto_fase, cambios_fase,
            ),
        ),
        (
            "INFORME 4 — DEMANDA CONSOLIDADA POR CLIENTE",
            lambda: ae.componer_informe_demanda_cliente(
                cliente_demanda, demanda_simulada,
            ),
        ),
    ]

    for label, componer in pruebas:
        print(f"\n=== {label} ===")
        try:
            subject, html, text = componer()
            subject_test = f"[PRUEBA] {subject}"
            ok, msg = ae.enviar_email(
                cfg, [destinatario_prueba], subject_test, html, text,
                con_bcc_admin=False,
            )
            if ok:
                print(f"✅ Enviado: {subject_test}")
            else:
                print(f"❌ Falló: {msg}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n💌 Revisá tu inbox en {destinatario_prueba}")
    print("    Vas a recibir 3 emails con prefijo [PRUEBA] en el subject.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
