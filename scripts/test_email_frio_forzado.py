#!/usr/bin/env python3
"""
Test de email + WhatsApp con escenario FORZADO de frío.

Sirve para validar que el bloque de impacto productivo cuantificado
aparece en el mail cuando hay frío suficiente, sin tener que esperar a
un frente real. El script:

  1. Lee tu cliente y lote reales de la DB (con sus overrides si están).
  2. Inyecta un clima sintético de frío serio (T° mín / viento / HR
     configurables vía flags).
  3. Construye las alertas sintéticas del evento.
  4. Llama a las MISMAS funciones que usa el cron diario
     (componer_alerta_diaria, generar_acciones_llm, generar_whatsapp_llm)
     con el clima forzado.
  5. Muestra en consola el subject + texto plano del mail (para revisar
     antes de enviar).
  6. Con --enviar, manda el email al destinatario real del cliente.
  7. Con --enviar --wa, además genera y muestra la frase del WhatsApp.

Uso:
    .venv/bin/python scripts/test_email_frio_forzado.py
    .venv/bin/python scripts/test_email_frio_forzado.py --enviar
    .venv/bin/python scripts/test_email_frio_forzado.py --enviar --wa
    .venv/bin/python scripts/test_email_frio_forzado.py --tmin -2 \\
        --viento 30 --hr 92 --dias 3   # frío crítico con barro
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from src import database as db
from src import alertas_email as ae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cliente", default="Mauricio Suarez",
        help="Nombre (o substring) del cliente a usar.",
    )
    parser.add_argument(
        "--enviar", action="store_true",
        help="Enviar el email real al destinatario del cliente. Si no "
              "se pasa, solo muestra el preview en consola.",
    )
    parser.add_argument(
        "--wa", action="store_true",
        help="Además generar la frase del WhatsApp (solo se imprime en "
              "consola, no se envía por sandbox).",
    )
    parser.add_argument("--tmin", type=float, default=1.0,
                         help="T° mínima del evento (°C). Default: 1.")
    parser.add_argument("--viento", type=float, default=25.0,
                         help="Viento máximo (km/h). Default: 25.")
    parser.add_argument("--hr", type=float, default=88.0,
                         help="HR máxima (%). Default: 88.")
    parser.add_argument("--lluvia", type=float, default=0.0,
                         help="Lluvia (mm). >5 activa pelaje mojado.")
    parser.add_argument("--barro", action="store_true",
                         help="Marcar barro en el contexto del evento.")
    parser.add_argument("--dias", type=int, default=2,
                         help="Días del evento. Default: 2.")
    parser.add_argument("--dias-previos", type=int, default=0,
                         help="Días previos con alerta (etapa). "
                               "Default: 0 (inicio).")
    args = parser.parse_args()

    db.init_db()

    # ── 1. Cliente real ────────────────────────────────────────
    clientes = db.listar_clientes()
    cliente = next(
        (c for c in clientes if args.cliente.lower() in c["nombre"].lower()),
        None,
    )
    if not cliente:
        print(f"✗ Cliente '{args.cliente}' no encontrado.")
        sys.exit(1)

    # ── 2. Lote activo con peso ────────────────────────────────
    lotes = db.listar_lotes(cliente_id=cliente["id"], estado="activo")
    lotes_con_peso = [
        l for l in lotes
        if (l.get("ultimo_peso_kg") or l.get("peso_ingreso_kg"))
    ]
    if not lotes_con_peso:
        print(f"✗ Cliente '{cliente['nombre']}' no tiene lotes activos "
              "con peso cargado.")
        sys.exit(1)
    lote = lotes_con_peso[0]
    peso = lote.get("ultimo_peso_kg") or lote.get("peso_ingreso_kg")

    print()
    print("=" * 72)
    print("ESCENARIO DE TEST")
    print("=" * 72)
    print(f"  Cliente: {cliente['nombre']} "
          f"({cliente.get('establecimiento') or cliente.get('localidad', '')})")
    print(f"  Lote: {lote['identificador']} · "
          f"{lote.get('categoria')} {lote.get('raza')} · "
          f"{peso:.0f} kg · {lote.get('cantidad_inicial')} cab.")
    if lote.get("adpv_objetivo_kg") or lote.get("energia_dieta_mcal_em_kg_ms"):
        print(f"  Override de lote: ADPV={lote.get('adpv_objetivo_kg')} "
              f"kg/día · Energía={lote.get('energia_dieta_mcal_em_kg_ms')} "
              "Mcal EM/kg MS")
    print(f"  Clima FORZADO: T° mín {args.tmin}°C · "
          f"viento {args.viento} km/h · HR {args.hr}% · "
          f"lluvia {args.lluvia} mm · "
          f"{'barro' if args.barro else 'piso seco'} · "
          f"{args.dias} días consecutivos")
    print(f"  Etapa: {'inicio' if args.dias_previos == 0 else f'persistencia (día {args.dias_previos + 1})'}")
    print()

    # ── 3. Construir clima_actual y alertas sintéticas ─────────
    clima_actual = {
        "temp_c": args.tmin + 3,
        "temp_min": args.tmin,
        "humedad_pct": args.hr,
        "viento_kmh": args.viento,
        "lluvia_mm": args.lluvia,
        "thi": 50,  # bajo (es evento de frío)
    }
    fecha_evt = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Severidad/nivel productivo derivados del escenario
    agravantes = (args.viento >= 20) + (args.hr >= 85) + args.barro + \
                 ((args.lluvia or 0) > 5)
    if args.tmin <= 0 and agravantes >= 3:
        severidad, nivel = "critica", "critico"
    elif args.tmin <= 5:
        severidad, nivel = "warning", "operativo"
    else:
        severidad, nivel = "warning", "moderado"

    descripcion = (
        f"T° mínima {args.tmin:.0f}°C, HR {args.hr:.0f}%, "
        f"viento {args.viento:.0f} km/h"
    )
    if args.lluvia > 0:
        descripcion += f", lluvia {args.lluvia:.0f} mm"
    if args.barro:
        descripcion += ", barro probable"
    descripcion += f". {args.dias} días consecutivos."

    accion_inicial = (
        "**📌 RESUMEN OPERATIVO**\n"
        f"📅 Cuándo: mañana ({fecha_evt})\n"
        f"{descripcion}\n"
        "\n"
        "**⚡ ACCIONES CLAVE**\n"
        "• Verificar reparos disponibles\n"
        "• Revisar acceso al agua\n"
        "• Asegurar mezcla cubierta"
    )

    alertas_por_lote = [{
        "lote": lote["identificador"],
        "categoria": lote.get("categoria", ""),
        "raza": lote.get("raza", ""),
        "peso_promedio_kg": peso,
        "cantidad_animales": lote.get("cantidad_inicial"),
        "adpv_objetivo_kg": lote.get("adpv_objetivo_kg"),
        "energia_dieta_mcal_em_kg_ms": lote.get(
            "energia_dieta_mcal_em_kg_ms"
        ),
        "alertas": [{
            "tipo": "frio",
            "nivel": nivel,
            "severidad": severidad,
            "titulo": f"Estrés por FRÍO {nivel.upper()}",
            "descripcion": descripcion,
            "accion": accion_inicial,
            "acciones": [],
            "_contexto": {
                "fecha": fecha_evt,
                "t_min": args.tmin,
                "temp_max": args.tmin + 5,
                "viento_kmh": args.viento,
                "humedad_pct": args.hr,
                "lluvia_mm": args.lluvia,
                "barro": args.barro,
            },
        }],
    }]

    # ── 4. Componer email (mismo path que el cron) ─────────────
    print("→ Componiendo email con escenario forzado…")
    print()
    subject, html, text = ae.componer_alerta_diaria(
        cliente=cliente,
        alertas_por_lote=alertas_por_lote,
        clima_actual=clima_actual,
        etapa="inicio" if args.dias_previos == 0 else "persistencia",
        dias_alerta_previos=args.dias_previos,
    )

    print("=" * 72)
    print(f"SUBJECT: {subject}")
    print("=" * 72)
    print()
    print(text)
    print()
    print("=" * 72)
    print(f"(HTML: {len(html):,} chars)")
    print("=" * 72)

    # ── 5. WhatsApp opcional (solo mostrar la frase) ───────────
    if args.wa:
        from src.ai_analisis_semanal import generar_whatsapp_llm
        from src.impacto_productivo import (
            estimar_impacto_frio, formato_impacto_texto,
        )
        imp = estimar_impacto_frio(
            peso_kg=peso,
            categoria=lote.get("categoria", ""),
            raza=lote.get("raza", ""),
            t_min_c=args.tmin, viento_kmh=args.viento, humedad_pct=args.hr,
            barro=args.barro,
            pelaje_mojado=(args.lluvia > 5),
            dias_evento=args.dias,
            cantidad=lote.get("cantidad_inicial"),
            adpv_objetivo_kg=lote.get("adpv_objetivo_kg"),
            energia_dieta_mcal_em_kg_ms=lote.get(
                "energia_dieta_mcal_em_kg_ms"
            ),
        )
        imp_txt = formato_impacto_texto(imp) if imp else None
        print()
        print("=" * 72)
        print("WHATSAPP — frase generada por LLM")
        print("=" * 72)
        if imp_txt:
            print(f"  Bloque de impacto inyectado:")
            print(f"  {imp_txt}")
            print()
        frase = generar_whatsapp_llm(
            cliente=cliente, tipo="frio", nivel=nivel,
            clima={
                "temperatura": args.tmin + 3,
                "min_nocturna": args.tmin,
                "viento_kmh": args.viento,
                "lluvia_mm": args.lluvia,
                "humedad_pct": args.hr,
                "thi": 50,
            },
            categoria=lote.get("categoria", ""),
            etapa="inicio" if args.dias_previos == 0 else "persistencia",
            dias_alerta_previos=args.dias_previos,
            ocurre_hoy=False, dias_hasta_evento=1,
            impacto_productivo_txt=imp_txt,
        )
        print(f"  → {frase or '(LLM no devolvió frase)'}")

    # ── 6. Envío real (si --enviar) ────────────────────────────
    if not args.enviar:
        print()
        print("→ Modo PREVIEW (no se envió nada). Agregá --enviar para "
              "mandar el email real al destinatario.")
        return

    cfg = ae.cargar_config_smtp()
    if not cfg:
        print("✗ No hay config SMTP cargada. Configurá SMTP en la "
              "pestaña Configuración primero.")
        sys.exit(1)

    destinatarios = db.listar_destinatarios(cliente)
    emails = [d["email"] for d in destinatarios if d.get("email")]
    if not emails:
        print(f"✗ Cliente '{cliente['nombre']}' no tiene destinatarios "
              "de email.")
        sys.exit(1)

    print()
    print(f"→ Enviando email a: {', '.join(emails)}")
    ok, msg = ae.enviar_email(cfg, emails, subject, html, text)
    if ok:
        print(f"  ✓ Email enviado. Revisá la casilla.")
    else:
        print(f"  ✗ Error: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
