"""
Base de datos local SQLite — trazabilidad de clientes, lotes, pesadas y dietas.

Estructura relacional:

    Cliente (1) ─── (N) Lote (1) ─── (N) Pesada
                              └────── (N) Dieta
                              └────── (N) Movimiento

Cada lote es una unidad de seguimiento (un grupo de animales en un corral).
Las pesadas y dietas son eventos temporales asociados a ese lote.

Archivo único: data/cattle_tracker.db (no requiere setup, sqlite3 viene con Python)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = Path("data/cattle_tracker.db")


# =====================================================================
# ESQUEMA
# =====================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS clientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL UNIQUE,
    contacto TEXT,
    establecimiento TEXT,
    localidad TEXT,
    lat REAL,
    lon REAL,
    email TEXT,
    alertas_email_activas INTEGER DEFAULT 1,
    whatsapp TEXT,
    alertas_whatsapp_activas INTEGER DEFAULT 1,
    bienvenida_email_enviada INTEGER DEFAULT 0,
    bienvenida_whatsapp_enviada INTEGER DEFAULT 0,
    notas TEXT,
    estado TEXT DEFAULT 'activo',
    fecha_baja TEXT,
    motivo_baja TEXT,
    fecha_alta TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contactos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER NOT NULL,
    nombre TEXT NOT NULL,
    rol TEXT,
    email TEXT,
    whatsapp TEXT,
    alertas_email_activas INTEGER DEFAULT 1,
    alertas_whatsapp_activas INTEGER DEFAULT 1,
    bienvenida_email_enviada INTEGER DEFAULT 0,
    bienvenida_whatsapp_enviada INTEGER DEFAULT 0,
    notas TEXT,
    fecha_alta TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cliente_id) REFERENCES clientes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_contactos_cliente ON contactos(cliente_id);

CREATE TABLE IF NOT EXISTS lotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER NOT NULL,
    identificador TEXT NOT NULL,
    corral TEXT,
    raza TEXT,
    categoria TEXT,
    fecha_ingreso TEXT,
    fecha_salida TEXT,
    cantidad_inicial INTEGER,
    peso_ingreso_kg REAL,
    objetivo_peso_kg REAL,
    objetivo_fecha TEXT,
    estado TEXT DEFAULT 'activo',
    notas TEXT,
    -- Override del cálculo de impacto productivo (NRC/NASEM). Si están
    -- en NULL, el módulo impacto_productivo usa los defaults de la
    -- categoría. Si los cargás acá, refinan el cálculo al sistema real
    -- del lote.
    adpv_objetivo_kg REAL,
    energia_dieta_mcal_em_kg_ms REAL,
    -- Sistema de entrega de comida — afecta qué tan rápido se pueden
    -- ajustar las recomendaciones nutricionales ante un evento climático.
    -- tipo_comedero_concentrado: 'lineal' (comedero lineal con carga
    -- desde mixer), 'silocomedero' (mezcla cargada en silocomedero,
    -- dura varios días), 'autoconsumo' (autoconsumo de concentrado).
    -- forraje_modalidad: 'mezclado' (rollo/silo va en el mixer junto
    -- a la ración) o 'aparte' (rollo en corral aparte, silo de
    -- autoconsumo separado, etc.).
    -- frecuencia_mezcla_dias: cada cuántos días se prepara/repone la
    -- mezcla. Lineal diario=1; mixer cada 2-3 días; silocomedero 4-5.
    tipo_comedero_concentrado TEXT,
    forraje_modalidad TEXT,
    frecuencia_mezcla_dias INTEGER,
    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cliente_id) REFERENCES clientes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pesadas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lote_id INTEGER NOT NULL,
    fecha TEXT NOT NULL,
    metodo TEXT,
    cantidad_animales INTEGER,
    peso_promedio_kg REAL,
    peso_total_kg REAL,
    desvio_kg REAL,
    cv_pct REAL,
    pesos_individuales_json TEXT,
    video_path TEXT,
    notas TEXT,
    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lote_id) REFERENCES lotes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dietas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lote_id INTEGER NOT NULL,
    fecha TEXT NOT NULL,
    composicion_json TEXT,
    costo_dia REAL,
    pb_pct REAL,
    em_mcal_dia REAL,
    consumo_ms_kg REAL,
    nnp_pct REAL,
    observaciones TEXT,
    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lote_id) REFERENCES lotes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS movimientos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lote_id INTEGER NOT NULL,
    fecha TEXT NOT NULL,
    tipo TEXT NOT NULL,
    cantidad INTEGER DEFAULT 0,
    detalles TEXT,
    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lote_id) REFERENCES lotes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entregas_producto (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER NOT NULL,
    lote_id INTEGER,
    producto_nombre TEXT NOT NULL,
    formato TEXT,
    cantidad_bolsas REAL DEFAULT 0,
    kg_por_bolsa REAL DEFAULT 30,
    kg_total REAL NOT NULL,
    fecha_entrega TEXT NOT NULL,
    precio_kg REAL DEFAULT 0,
    precio_total REAL DEFAULT 0,
    notas TEXT,
    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cliente_id) REFERENCES clientes(id) ON DELETE CASCADE,
    FOREIGN KEY (lote_id) REFERENCES lotes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_lotes_cliente ON lotes(cliente_id);
CREATE INDEX IF NOT EXISTS idx_pesadas_lote ON pesadas(lote_id, fecha);
CREATE INDEX IF NOT EXISTS idx_dietas_lote ON dietas(lote_id, fecha);
CREATE INDEX IF NOT EXISTS idx_entregas_cliente
    ON entregas_producto(cliente_id, fecha_entrega);
CREATE INDEX IF NOT EXISTS idx_entregas_lote
    ON entregas_producto(lote_id, fecha_entrega);

-- Blob-cache de vistas precomputadas del dashboard. Un cron externo
-- (GitHub Actions cada 5 min) rellena esta tabla con el resultado
-- serializado del bloque de logística — la app solo lo lee (1 query)
-- en lugar de recalcular ~50 queries × 20 clientes cada visita.
CREATE TABLE IF NOT EXISTS dashboard_cache (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


# =====================================================================
# CONEXIÓN
# =====================================================================
# NOTA: get_conn() ahora delega en src/db_backend.py, que decide entre
# SQLite y Postgres (Supabase) según la variable de entorno DATABASE_URL.
#
# - Sin DATABASE_URL → SQLite local (comportamiento original, default).
# - Con DATABASE_URL → Postgres. El backend traduce automáticamente
#   los placeholders `?` → `%s` y adapta el resto de diferencias.
#
# El resto del código NO cambia: sigue haciendo `with get_conn() as
# conn: conn.execute("... WHERE x = ?", (val,))`.
#
# Rollback plan: borrás DATABASE_URL del .env → vuelve todo a SQLite.

from src.db_backend import (
    get_conn as _get_conn_backend,
    usando_postgres as _usando_postgres,
    backend_activo as _backend_activo,
)


@contextmanager
def get_conn(db_path: Optional[Path] = None):
    """Context manager que devuelve la conexión activa (SQLite o Postgres)."""
    with _get_conn_backend(db_path) as conn:
        yield conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Crea las tablas si no existen + aplica migraciones.

    En Postgres saltea TODO (incluido executescript). Las tablas ya
    fueron creadas con scripts/migrar_sqlite_a_supabase.py. Correr el
    SCHEMA de SQLite contra Postgres tira SyntaxError porque los tipos
    y sintaxis difieren aunque tengan IF NOT EXISTS.
    """
    # Postgres: no correr executescript ni migraciones inline (ambos
    # usan sintaxis SQLite-only). Salir temprano.
    if _usando_postgres():
        return
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        # Migraciones: agregar columnas que pueden faltar en DBs viejas
        try:
            existing_cols = {row["name"] for row in conn.execute(
                "PRAGMA table_info(clientes)"
            ).fetchall()}
            if "lat" not in existing_cols:
                conn.execute("ALTER TABLE clientes ADD COLUMN lat REAL")
            if "lon" not in existing_cols:
                conn.execute("ALTER TABLE clientes ADD COLUMN lon REAL")
            if "email" not in existing_cols:
                conn.execute("ALTER TABLE clientes ADD COLUMN email TEXT")
            if "alertas_email_activas" not in existing_cols:
                conn.execute(
                    "ALTER TABLE clientes ADD COLUMN "
                    "alertas_email_activas INTEGER DEFAULT 1"
                )
            if "whatsapp" not in existing_cols:
                conn.execute("ALTER TABLE clientes ADD COLUMN whatsapp TEXT")
            if "alertas_whatsapp_activas" not in existing_cols:
                conn.execute(
                    "ALTER TABLE clientes ADD COLUMN "
                    "alertas_whatsapp_activas INTEGER DEFAULT 1"
                )
            # Flags de bienvenida — para que el contacto principal reciba el
            # mensaje explicativo la primera vez que llega una alerta.
            agregue_bienvenida_em = (
                "bienvenida_email_enviada" not in existing_cols
            )
            agregue_bienvenida_wa = (
                "bienvenida_whatsapp_enviada" not in existing_cols
            )
            if agregue_bienvenida_em:
                conn.execute(
                    "ALTER TABLE clientes ADD COLUMN "
                    "bienvenida_email_enviada INTEGER DEFAULT 0"
                )
            if agregue_bienvenida_wa:
                conn.execute(
                    "ALTER TABLE clientes ADD COLUMN "
                    "bienvenida_whatsapp_enviada INTEGER DEFAULT 0"
                )
            # Si la columna recién se creó, marcamos como "ya recibieron"
            # a los clientes existentes que ya tienen email/whatsapp cargados.
            # Justificación: ya están recibiendo alertas — no tiene sentido
            # mandarles una bienvenida retroactiva.
            if agregue_bienvenida_em:
                conn.execute(
                    "UPDATE clientes SET bienvenida_email_enviada = 1 "
                    "WHERE email IS NOT NULL AND email != ''"
                )
            if agregue_bienvenida_wa:
                conn.execute(
                    "UPDATE clientes SET bienvenida_whatsapp_enviada = 1 "
                    "WHERE whatsapp IS NOT NULL AND whatsapp != ''"
                )
            # Estado explícito del cliente (activo/baja) — independiente
            # de los toggles de alertas. Un cliente puede estar ACTIVO
            # sin recibir alertas (eligió no recibirlas), o de BAJA (ya
            # no es cliente y se archivó).
            if "estado" not in existing_cols:
                conn.execute(
                    "ALTER TABLE clientes ADD COLUMN "
                    "estado TEXT DEFAULT 'activo'"
                )
                # Marcar como activos a todos los existentes
                conn.execute(
                    "UPDATE clientes SET estado = 'activo' "
                    "WHERE estado IS NULL OR estado = ''"
                )
            if "fecha_baja" not in existing_cols:
                conn.execute(
                    "ALTER TABLE clientes ADD COLUMN fecha_baja TEXT"
                )
            if "motivo_baja" not in existing_cols:
                conn.execute(
                    "ALTER TABLE clientes ADD COLUMN motivo_baja TEXT"
                )
        except Exception:
            pass

        # Migración tabla `lotes` — campos para refinar el cálculo de
        # impacto productivo NRC con datos reales del lote (override de
        # los defaults por categoría).
        try:
            cols_lotes = {row["name"] for row in conn.execute(
                "PRAGMA table_info(lotes)"
            ).fetchall()}
            if "adpv_objetivo_kg" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN adpv_objetivo_kg REAL"
                )
            if "energia_dieta_mcal_em_kg_ms" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "energia_dieta_mcal_em_kg_ms REAL"
                )
            # Sistema de entrega de comida — define qué tan rápido se
            # puede ajustar la mezcla ante un evento climático.
            if "tipo_comedero_concentrado" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "tipo_comedero_concentrado TEXT"
                )
            if "forraje_modalidad" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "forraje_modalidad TEXT"
                )
            if "frecuencia_mezcla_dias" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "frecuencia_mezcla_dias INTEGER"
                )
            # --- Encargado/operario del lote ---
            # Persona que efectivamente prepara la mezcla y la carga al
            # comedero todos los días. Puede ser distinta del productor
            # (dueño). Se le manda WhatsApp 17:00 con un link a un mini-
            # form donde ingresa los kg cargados de cada ingrediente.
            if "encargado_nombre" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "encargado_nombre TEXT"
                )
            if "encargado_whatsapp" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "encargado_whatsapp TEXT"
                )
            if "carga_diaria_activa" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "carga_diaria_activa INTEGER DEFAULT 0"
                )
            # Horarios configurables de las comidas del día. Cuando
            # se activa la carga diaria, el cron manda WhatsApp en
            # esos horarios (±10 min). Si cant_comidas_diarias=1
            # solo usa hora_comida_1.
            if "cant_comidas_diarias" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "cant_comidas_diarias INTEGER DEFAULT 1"
                )
            if "hora_comida_1" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "hora_comida_1 TEXT DEFAULT '08:30'"
                )
            if "hora_comida_2" not in cols_lotes:
                conn.execute(
                    "ALTER TABLE lotes ADD COLUMN "
                    "hora_comida_2 TEXT DEFAULT '16:00'"
                )
        except Exception:
            pass

        # Tabla pedidos_carga_enviados: dedup del cron para evitar
        # mandar el mismo WhatsApp dos veces. Una fila por
        # (lote, fecha, comida_n, intento_n).
        # intento_n=1: envío inicial a la hora configurada.
        # intento_n=2: recordatorio 1 hora después si no cargó.
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pedidos_carga_enviados (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lote_id INTEGER NOT NULL,
                    fecha TEXT NOT NULL,
                    comida_n INTEGER NOT NULL,
                    intento_n INTEGER NOT NULL DEFAULT 1,
                    fecha_envio TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (lote_id, fecha, comida_n, intento_n),
                    FOREIGN KEY (lote_id) REFERENCES lotes(id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_pedidos_carga_lote_fecha "
                "ON pedidos_carga_enviados(lote_id, fecha)"
            )
        except Exception:
            pass

        # Migración tabla `movimientos` — registro histórico de cambios
        # en la cantidad de animales del lote (muerte, venta, traslado,
        # ingreso). La cantidad vigente del lote en cualquier fecha se
        # calcula como cantidad_inicial + suma(movimientos hasta esa fecha).
        # Campos nuevos:
        #   - kg_promedio_animal: peso por animal al momento del movimiento
        #     (útil en ventas para calcular kg vendidos, y en muertes para
        #     saber a qué edad/peso se perdió).
        #   - destino_lote_id: en traslados, el lote al que va el animal.
        # El campo `tipo` define el motivo: 'muerte', 'venta',
        # 'traslado_egreso', 'traslado_ingreso', 'ingreso'. La `cantidad`
        # se guarda siempre positiva; el signo se deriva del tipo.
        try:
            cols_mov = {row["name"] for row in conn.execute(
                "PRAGMA table_info(movimientos)"
            ).fetchall()}
            if "kg_promedio_animal" not in cols_mov:
                conn.execute(
                    "ALTER TABLE movimientos ADD COLUMN "
                    "kg_promedio_animal REAL"
                )
            if "destino_lote_id" not in cols_mov:
                conn.execute(
                    "ALTER TABLE movimientos ADD COLUMN "
                    "destino_lote_id INTEGER"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_movimientos_lote "
                "ON movimientos(lote_id, fecha)"
            )
        except Exception:
            pass

        # Tabla de cargas del silocomedero — cada fila es un día en que
        # el productor preparó la mezcla y la cargó en el silocomedero.
        # Permite proyectar cuándo se agota (kg cargados ÷ consumo diario
        # de la dieta vigente) y emitir la alerta 1 día antes.
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cargas_silocomedero (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lote_id INTEGER NOT NULL,
                    fecha_carga TEXT NOT NULL,
                    kg_cargados REAL NOT NULL,
                    detalles TEXT,
                    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (lote_id) REFERENCES lotes(id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cargas_silo "
                "ON cargas_silocomedero(lote_id, fecha_carga)"
            )
        except Exception:
            pass

        # Migración (idempotente): tabla cargas_silocomedero ahora se usa
        # también para comedero lineal (carga diaria) y por ingrediente.
        # - tipo_carga: 'silo_carga' (varios días) / 'lineal_diario' (1 día)
        # - desglose_ingredientes_json: si el productor cargó desglosado,
        #   JSON con lista de {nombre, kg}.
        # - dias_cubiertos: para silo, # días que dura esa carga; para
        #   lineal_diario es 1.
        for _alter in [
            "ALTER TABLE cargas_silocomedero ADD COLUMN tipo_carga TEXT "
            "DEFAULT 'silo_carga'",
            "ALTER TABLE cargas_silocomedero ADD COLUMN "
            "desglose_ingredientes_json TEXT",
            "ALTER TABLE cargas_silocomedero ADD COLUMN "
            "dias_cubiertos REAL DEFAULT 1",
            # Hora exacta de la carga (HH:MM). Permite registrar varias
            # cargas por día (ej. 2 comidas en comedero lineal) y ver
            # irregularidades de horario.
            "ALTER TABLE cargas_silocomedero ADD COLUMN "
            "hora_carga TEXT",
        ]:
            try:
                conn.execute(_alter)
            except Exception:
                pass

        # Tabla histórico de impactos productivos por lote.
        # Cada fila es un cálculo de impacto NRC para un lote en una
        # fecha. Permite reconstruir cronológicamente qué eventos
        # afectaron a cada lote, qué pérdidas se proyectaron y qué
        # se confirmó después con datos reales del clima.
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS impactos_lote (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lote_id INTEGER NOT NULL,
                    fecha_calculo TEXT NOT NULL,
                    fecha_inicio_evento TEXT,
                    fecha_fin_evento TEXT,
                    tipo_evento TEXT,
                    severidad TEXT,
                    gasto_extra_pct_min REAL,
                    gasto_extra_pct_max REAL,
                    adpv_perdida_min_kg REAL,
                    adpv_perdida_max_kg REAL,
                    kg_lote_total_min REAL,
                    kg_lote_total_max REAL,
                    pct_adpv_min REAL,
                    pct_adpv_max REAL,
                    dias_evento INTEGER,
                    cantidad_animales INTEGER,
                    peso_promedio_kg REAL,
                    clima_resumen_json TEXT,
                    notas TEXT,
                    estado TEXT DEFAULT 'proyectado',
                    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (lote_id) REFERENCES lotes(id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_impactos_lote "
                "ON impactos_lote(lote_id, fecha_calculo)"
            )
            # Migración: si la tabla existía sin columna estado,
            # agregarla con default 'proyectado' (registros viejos).
            cols_imp = {row["name"] for row in conn.execute(
                "PRAGMA table_info(impactos_lote)"
            ).fetchall()}
            if "estado" not in cols_imp:
                conn.execute(
                    "ALTER TABLE impactos_lote ADD COLUMN "
                    "estado TEXT DEFAULT 'proyectado'"
                )
                conn.execute(
                    "UPDATE impactos_lote SET estado='proyectado' "
                    "WHERE estado IS NULL"
                )
        except Exception:
            pass

        # Tabla de fotos de inspección del lote — asociadas a una
        # consulta (recordatorios_llamada) y al lote. Cada foto tiene
        # tipo (bosta/animales/comedero/corral/bebedero/otros) y un
        # comentario opcional. El archivo físico vive en disco bajo
        # data/fotos_lote/<lote_id>/<recordatorio_id>/.
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fotos_lote (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lote_id INTEGER NOT NULL,
                    recordatorio_id INTEGER,
                    tipo TEXT NOT NULL,
                    archivo_path TEXT NOT NULL,
                    comentario TEXT,
                    fecha TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (lote_id) REFERENCES lotes(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (recordatorio_id)
                        REFERENCES recordatorios_llamada(id)
                        ON DELETE SET NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fotos_lote "
                "ON fotos_lote(lote_id, recordatorio_id, fecha)"
            )
        except Exception:
            pass

        # Tabla de log de envíos de email (idempotencia diaria + auditoría)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alertas_enviadas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TEXT NOT NULL,
                    cliente_id INTEGER,
                    destinatario TEXT,
                    asunto TEXT,
                    n_alertas INTEGER DEFAULT 0,
                    estado TEXT,
                    error TEXT,
                    tipo TEXT DEFAULT 'diaria',
                    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migración: agregar columna tipo si no existe
            cols_alertas = {row["name"] for row in conn.execute(
                "PRAGMA table_info(alertas_enviadas)"
            ).fetchall()}
            if "tipo" not in cols_alertas:
                conn.execute(
                    "ALTER TABLE alertas_enviadas ADD COLUMN "
                    "tipo TEXT DEFAULT 'diaria'"
                )
            # Migración: agregar `lectura_tecnica` para guardar el análisis
            # generado por el LLM. Lo usamos como MEMORIA: cuando el LLM
            # arme la próxima lectura, puede ver las anteriores y evitar
            # repetir frases/ángulos → mantiene la atención del productor.
            if "lectura_tecnica" not in cols_alertas:
                conn.execute(
                    "ALTER TABLE alertas_enviadas ADD COLUMN "
                    "lectura_tecnica TEXT"
                )
        except Exception:
            pass

        # Tabla de snapshots del pronóstico semanal (para comparar miércoles
        # vs lunes y detectar cambios significativos en el pronóstico).
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pronosticos_semanales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cliente_id INTEGER NOT NULL,
                    fecha_lunes TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (cliente_id) REFERENCES clientes(id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pron_sem "
                "ON pronosticos_semanales(cliente_id, fecha_lunes)"
            )
        except Exception:
            pass

        # Tabla de log de envíos de WhatsApp (dedup por clave + ventana 12hs)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alertas_whatsapp_enviadas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cliente_id INTEGER,
                    destinatario TEXT,
                    clave_dedup TEXT,
                    mensaje TEXT,
                    estado TEXT,
                    error TEXT,
                    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wapp_dedup "
                "ON alertas_whatsapp_enviadas(clave_dedup, fecha_creacion)"
            )
        except Exception:
            pass

        # Tabla de categorías de animales — administrada por el usuario.
        # Reemplaza la lista hardcodeada en los selectbox de Nuevo Lote,
        # Drone y Asesor IA. Permite renombrar, agregar y borrar.
        # `adpv_default_kg_dia` se usa cuando el lote no tiene ADPV
        # propio cargado (ver impacto_productivo.DEFAULTS_CATEGORIA).
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS categorias_animales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT NOT NULL UNIQUE,
                    adpv_default_kg_dia REAL DEFAULT 0,
                    orden INTEGER DEFAULT 100,
                    activo INTEGER DEFAULT 1,
                    notas TEXT DEFAULT '',
                    fecha_creacion TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Seed inicial — solo se inserta si la tabla está vacía
            cnt = conn.execute(
                "SELECT COUNT(*) AS n FROM categorias_animales"
            ).fetchone()["n"]
            if cnt == 0:
                _seed_cats = [
                    ("ternero",      0.80, 10, "Macho destetado, < 200 kg"),
                    ("ternera",      0.70, 15, "Hembra destetada, < 200 kg"),
                    ("recria",       1.00, 20, "Etapa entre destete e ingreso a terminación"),
                    ("vaquillona",   0.90, 30, "Hembra joven, sin parir"),
                    ("novillito",    1.10, 40, "Macho castrado 200-330 kg"),
                    ("novillo",      1.20, 50, "Macho castrado, terminación"),
                    ("vaca_adulta",  0.40, 60, "Vaca de cría o refugo"),
                    ("toro",         0.50, 70, "Reproductor"),
                ]
                conn.executemany(
                    "INSERT INTO categorias_animales "
                    "(nombre, adpv_default_kg_dia, orden, notas) "
                    "VALUES (?, ?, ?, ?)",
                    _seed_cats,
                )
        except Exception:
            pass

        # ────────────────────────────────────────────────────────
        # RECORDATORIOS DE LLAMADA AL CLIENTE
        # Manual + sugerencias automáticas en momentos clave
        # (lote nuevo, cambio de fase próximo, clima crítico,
        # sin contacto hace > 30 días).
        # ────────────────────────────────────────────────────────
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recordatorios_llamada (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cliente_id INTEGER NOT NULL,
                    fecha_objetivo TEXT NOT NULL,
                    motivo TEXT DEFAULT '',
                    origen TEXT DEFAULT 'manual',
                    estado TEXT DEFAULT 'pendiente',
                    notas_cierre TEXT DEFAULT '',
                    evaluacion_json TEXT DEFAULT '',
                    lote_id INTEGER,
                    creado_en TEXT DEFAULT CURRENT_TIMESTAMP,
                    completado_en TEXT DEFAULT NULL,
                    FOREIGN KEY (cliente_id)
                        REFERENCES clientes(id) ON DELETE CASCADE
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_recordatorios_pendientes "
                "ON recordatorios_llamada(estado, fecha_objetivo)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_recordatorios_cliente "
                "ON recordatorios_llamada(cliente_id, estado)"
            )
            # Migración para DBs viejas: agregar columnas si faltan
            for col_name, col_def in (
                ("evaluacion_json", "TEXT DEFAULT ''"),
                ("lote_id", "INTEGER"),
            ):
                try:
                    conn.execute(
                        f"ALTER TABLE recordatorios_llamada "
                        f"ADD COLUMN {col_name} {col_def}"
                    )
                except Exception:
                    pass  # ya existe
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_recordatorios_lote "
                "ON recordatorios_llamada(lote_id, estado)"
            )
        except Exception:
            pass


# =====================================================================
# CATEGORÍAS DE ANIMALES (CRUD)
# =====================================================================

def listar_categorias(solo_activas: bool = True) -> list:
    """Devuelve la lista de categorías ordenadas por `orden`.

    Si `solo_activas` está en True (default) sólo trae las que tienen
    activo=1. Para la pantalla de administración usar
    `solo_activas=False` para mostrar también las desactivadas.
    """
    with get_conn() as conn:
        sql = "SELECT * FROM categorias_animales"
        if solo_activas:
            sql += " WHERE activo = 1"
        sql += " ORDER BY orden, nombre"
        return [dict(r) for r in conn.execute(sql).fetchall()]


def nombres_categorias(solo_activas: bool = True) -> list:
    """Atajo para los selectbox — devuelve sólo los nombres."""
    return [c["nombre"] for c in listar_categorias(solo_activas)]


def crear_categoria(nombre: str, adpv_default_kg_dia: float = 0.0,
                    orden: int = 100, notas: str = "") -> int:
    """Crea una categoría nueva. Devuelve el id creado.

    Lanza ValueError si el nombre ya existe (la tabla tiene UNIQUE).
    """
    nombre = (nombre or "").strip().lower()
    if not nombre:
        raise ValueError("El nombre no puede estar vacío.")
    with get_conn() as conn:
        ya = conn.execute(
            "SELECT id FROM categorias_animales WHERE nombre = ?",
            (nombre,),
        ).fetchone()
        if ya:
            raise ValueError(
                f"Ya existe una categoría con el nombre '{nombre}'."
            )
        cur = conn.execute(
            "INSERT INTO categorias_animales "
            "(nombre, adpv_default_kg_dia, orden, notas) "
            "VALUES (?, ?, ?, ?)",
            (nombre, float(adpv_default_kg_dia), int(orden), notas),
        )
        return cur.lastrowid


def actualizar_categoria(categoria_id: int, nombre: str = None,
                         adpv_default_kg_dia: float = None,
                         orden: int = None, activo: int = None,
                         notas: str = None) -> None:
    """Actualiza los campos no-None de la categoría."""
    sets, params = [], []
    if nombre is not None:
        sets.append("nombre = ?")
        params.append(nombre.strip().lower())
    if adpv_default_kg_dia is not None:
        sets.append("adpv_default_kg_dia = ?")
        params.append(float(adpv_default_kg_dia))
    if orden is not None:
        sets.append("orden = ?")
        params.append(int(orden))
    if activo is not None:
        sets.append("activo = ?")
        params.append(int(activo))
    if notas is not None:
        sets.append("notas = ?")
        params.append(notas)
    if not sets:
        return
    params.append(categoria_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE categorias_animales SET {', '.join(sets)} "
            f"WHERE id = ?",
            params,
        )


def eliminar_categoria(categoria_id: int) -> None:
    """Borrado físico. Si la categoría está en uso por algún lote,
    es preferible desactivarla con activo=0 — esta función no
    valida referencias."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM categorias_animales WHERE id = ?",
            (categoria_id,),
        )


# =====================================================================
# CLIENTES
# =====================================================================

def crear_cliente(nombre: str, contacto: str = "", establecimiento: str = "",
                  localidad: str = "", notas: str = "",
                  email: str = "", alertas_email_activas: int = 1,
                  whatsapp: str = "",
                  alertas_whatsapp_activas: int = 1) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO clientes (nombre, contacto, establecimiento, localidad, "
            "notas, email, alertas_email_activas, whatsapp, alertas_whatsapp_activas) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (nombre, contacto, establecimiento, localidad, notas,
             email, alertas_email_activas,
             whatsapp, alertas_whatsapp_activas),
        )
        return cur.lastrowid


def registrar_alerta_enviada(fecha: str, cliente_id: Optional[int],
                              destinatario: str, asunto: str,
                              n_alertas: int, estado: str,
                              error: str = "",
                              tipo: str = "diaria",
                              lectura_tecnica: Optional[str] = None) -> None:
    """Loggea un envío de email para auditoría e idempotencia diaria.

    `tipo` distingue ventanas para que el cron de la tarde (tipo='tarde')
    no choque con el dedup de la mañana (tipo='diaria').

    `lectura_tecnica`: texto del análisis generado por el LLM (si lo hubo).
    Se guarda para usarlo como MEMORIA en el siguiente email — el LLM verá
    los textos recientes y evitará repetir frases / ángulos. Esto mantiene
    la atención del productor (anti banner-blindness).
    """
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO alertas_enviadas (fecha, cliente_id, destinatario, "
            "asunto, n_alertas, estado, error, tipo, lectura_tecnica) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fecha, cliente_id, destinatario, asunto, n_alertas,
             estado, error, tipo, lectura_tecnica),
        )


def obtener_lecturas_recientes(cliente_id: Optional[int],
                                tipo: Optional[str] = None,
                                limite: int = 3) -> List[str]:
    """Devuelve las últimas N lecturas técnicas enviadas a un cliente.

    Se usa como MEMORIA para el LLM: cuando arme el próximo análisis,
    puede ver lo que ya dijo y evitar repetir frases/ángulos.

    `tipo` filtra por canal (diaria/tarde/semanal). Si es None, mezcla
    todos los tipos (útil cuando el LLM debe variar entre formatos).
    """
    if cliente_id is None:
        return []
    sql = (
        "SELECT lectura_tecnica FROM alertas_enviadas "
        "WHERE cliente_id = ? AND lectura_tecnica IS NOT NULL "
        "AND TRIM(lectura_tecnica) != '' AND estado = 'enviada' "
    )
    params: List[object] = [cliente_id]
    if tipo:
        sql += "AND tipo = ? "
        params.append(tipo)
    sql += "ORDER BY id DESC LIMIT ?"
    params.append(int(limite))
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    out: List[str] = []
    for r in rows:
        txt = (r["lectura_tecnica"] if isinstance(r, sqlite3.Row)
               else r[0])
        if txt and str(txt).strip():
            out.append(str(txt).strip())
    return out


# ─────────────────────────────────────────────────────────────────────
# FOTOS DE INSPECCIÓN DEL LOTE
# Asociadas a una consulta (recordatorios_llamada). El archivo físico
# vive en data/fotos_lote/<lote_id>/<recordatorio_id>/.
# ─────────────────────────────────────────────────────────────────────

# Tipos válidos de foto. La UI muestra emoji + label legible.
TIPOS_FOTO_LOTE = {
    "bosta":     {"label": "Bosta",     "emoji": "🟫"},
    "animales":  {"label": "Animales",  "emoji": "🐄"},
    "comedero":  {"label": "Comedero",  "emoji": "🌾"},
    "corral":    {"label": "Corral",    "emoji": "🏗️"},
    "bebedero":  {"label": "Bebedero",  "emoji": "💧"},
    "otros":     {"label": "Otros",     "emoji": "📋"},
}


def registrar_foto_lote(lote_id: int, recordatorio_id: Optional[int],
                         tipo: str, archivo_path: str,
                         comentario: str = "",
                         fecha: Optional[str] = None) -> int:
    """Inserta un registro de foto. Devuelve el id."""
    if tipo not in TIPOS_FOTO_LOTE:
        tipo = "otros"
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO fotos_lote "
            "(lote_id, recordatorio_id, tipo, archivo_path, comentario, fecha) "
            "VALUES (?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
            (lote_id, recordatorio_id, tipo, archivo_path,
             comentario or "", fecha),
        )
        return cur.lastrowid


def listar_fotos_lote(lote_id: int,
                       recordatorio_id: Optional[int] = None) -> List[Dict]:
    """Lista fotos de un lote. Si pasás recordatorio_id, solo las de
    esa consulta. Si no, todas las del lote (para galería completa).
    """
    sql = "SELECT * FROM fotos_lote WHERE lote_id = ?"
    params: List[object] = [lote_id]
    if recordatorio_id is not None:
        sql += " AND recordatorio_id = ?"
        params.append(recordatorio_id)
    sql += " ORDER BY fecha DESC, id DESC"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def eliminar_foto_lote(foto_id: int, borrar_archivo: bool = True) -> bool:
    """Borra una foto de la DB. Si borrar_archivo=True, también borra
    el archivo del disco. Devuelve True si se borró algo.
    """
    with get_conn() as conn:
        r = conn.execute(
            "SELECT archivo_path FROM fotos_lote WHERE id = ?",
            (foto_id,),
        ).fetchone()
        if not r:
            return False
        archivo = r["archivo_path"] if isinstance(r, sqlite3.Row) else r[0]
        conn.execute("DELETE FROM fotos_lote WHERE id = ?", (foto_id,))
    if borrar_archivo and archivo:
        try:
            from pathlib import Path
            p = Path(archivo)
            if p.exists() and p.is_file():
                p.unlink()
        except Exception:
            pass
    return True


def actualizar_comentario_foto(foto_id: int, comentario: str) -> bool:
    """Editar el comentario de una foto sin re-subir el archivo."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE fotos_lote SET comentario = ? WHERE id = ?",
            (comentario or "", foto_id),
        )
        return cur.rowcount > 0


def alerta_ya_enviada_hoy(cliente_id: Optional[int], destinatario: str,
                           fecha: str, tipo: str = "diaria") -> bool:
    """Evita duplicar envíos si el cron corre dos veces el mismo día.

    `tipo` permite tener ventanas separadas: 'diaria' (8 AM) y 'tarde'
    (18 PM) coexisten sin pisarse el dedup.
    """
    with get_conn() as conn:
        r = conn.execute(
            "SELECT 1 FROM alertas_enviadas WHERE fecha = ? AND destinatario = ? "
            "AND (cliente_id = ? OR (cliente_id IS NULL AND ? IS NULL)) "
            "AND estado = 'enviada' AND tipo = ? LIMIT 1",
            (fecha, destinatario, cliente_id, cliente_id, tipo),
        ).fetchone()
        return r is not None


def registrar_whatsapp_enviado(cliente_id: Optional[int],
                                 destinatario: str, clave_dedup: str,
                                 mensaje: str, estado: str,
                                 error: str = "") -> None:
    """Loggea envío de WhatsApp para deduplicación + auditoría."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO alertas_whatsapp_enviadas "
            "(cliente_id, destinatario, clave_dedup, mensaje, estado, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cliente_id, destinatario, clave_dedup, mensaje[:500], estado, error),
        )


def whatsapp_ya_enviado(clave_dedup: str, ventana_horas: int = 12) -> bool:
    """¿Esta misma alerta ya se envió en las últimas N horas?

    Sirve para no spamear: si el clima sigue crítico hora tras hora, no
    repetimos la misma alerta cada vez que corre el cron.
    """
    with get_conn() as conn:
        r = conn.execute(
            f"SELECT 1 FROM alertas_whatsapp_enviadas "
            f"WHERE clave_dedup = ? AND estado = 'enviada' "
            f"AND fecha_creacion >= datetime('now', '-{int(ventana_horas)} hours') "
            f"LIMIT 1",
            (clave_dedup,),
        ).fetchone()
        return r is not None


def listar_clientes() -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM clientes ORDER BY nombre"
        ).fetchall()
        return [dict(r) for r in rows]


def obtener_cliente(cliente_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM clientes WHERE id = ?", (cliente_id,)).fetchone()
        return dict(r) if r else None


def dar_de_baja_cliente(
    cliente_id: int, motivo: str = "",
    desactivar_alertas: bool = True,
) -> None:
    """Marca a un cliente como dado de baja (estado='baja') y
    registra la fecha. Por default también apaga las alertas porque
    normalmente no querés seguir mandándole mensajes a un cliente que
    se dio de baja, pero podés pasar desactivar_alertas=False si
    querés mantenerlas activas (caso raro)."""
    from datetime import datetime as _dt
    campos = {
        "estado": "baja",
        "fecha_baja": _dt.now().strftime("%Y-%m-%d"),
        "motivo_baja": motivo or "",
    }
    if desactivar_alertas:
        campos["alertas_email_activas"] = 0
        campos["alertas_whatsapp_activas"] = 0
    actualizar_cliente(cliente_id, **campos)


def reactivar_cliente(cliente_id: int) -> None:
    """Vuelve a marcar a un cliente como activo. Limpia la fecha y
    motivo de baja. NO reactiva las alertas automáticamente — vos las
    podés volver a prender en la ficha si querés."""
    actualizar_cliente(
        cliente_id, estado="activo",
        fecha_baja=None, motivo_baja=None,
    )


def actualizar_cliente(cliente_id: int, **campos) -> None:
    if not campos:
        return
    sets = ", ".join(f"{k} = ?" for k in campos)
    valores = list(campos.values()) + [cliente_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE clientes SET {sets} WHERE id = ?", valores)


def eliminar_cliente(cliente_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM clientes WHERE id = ?", (cliente_id,))


# =====================================================================
# CONTACTOS (múltiples por cliente: productor + encargado + comedero, etc.)
# =====================================================================

def listar_contactos(cliente_id: int) -> List[Dict]:
    """Devuelve todos los contactos extra de un cliente (no incluye el principal)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM contactos WHERE cliente_id = ? ORDER BY fecha_alta",
            (cliente_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def crear_contacto(cliente_id: int, nombre: str, rol: str = "",
                   email: str = "", whatsapp: str = "",
                   alertas_email_activas: int = 1,
                   alertas_whatsapp_activas: int = 1,
                   notas: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO contactos (cliente_id, nombre, rol, email, whatsapp, "
            "alertas_email_activas, alertas_whatsapp_activas, notas) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cliente_id, nombre, rol, email, whatsapp,
             alertas_email_activas, alertas_whatsapp_activas, notas),
        )
        return cur.lastrowid


def obtener_contacto(contacto_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT * FROM contactos WHERE id = ?", (contacto_id,)
        ).fetchone()
        return dict(r) if r else None


def actualizar_contacto(contacto_id: int, **campos) -> None:
    if not campos:
        return
    sets = ", ".join(f"{k} = ?" for k in campos)
    valores = list(campos.values()) + [contacto_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE contactos SET {sets} WHERE id = ?", valores)


def eliminar_contacto(contacto_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM contactos WHERE id = ?", (contacto_id,))


def listar_destinatarios(cliente: Dict) -> List[Dict]:
    """Devuelve la lista unificada de destinatarios de un cliente:
    el contacto principal (de la tabla clientes) + los contactos extras
    (de la tabla contactos).

    Cada item es un dict con shape uniforme:
        {
            'origen': 'principal' | 'extra',
            'id': cliente_id (si principal) o contacto_id (si extra),
            'nombre': str,
            'rol': str,
            'email': str,
            'whatsapp': str,
            'alertas_email_activas': 0/1,
            'alertas_whatsapp_activas': 0/1,
            'bienvenida_email_enviada': 0/1,
            'bienvenida_whatsapp_enviada': 0/1,
        }
    """
    out = []
    # Principal — viene de la tabla clientes
    out.append({
        "origen": "principal",
        "id": cliente.get("id"),
        "nombre": cliente.get("contacto") or cliente.get("nombre", ""),
        "rol": "Productor",
        "email": (cliente.get("email") or "").strip(),
        "whatsapp": (cliente.get("whatsapp") or "").strip(),
        "alertas_email_activas": cliente.get("alertas_email_activas", 1),
        "alertas_whatsapp_activas": cliente.get("alertas_whatsapp_activas", 1),
        "bienvenida_email_enviada": cliente.get("bienvenida_email_enviada", 0),
        "bienvenida_whatsapp_enviada": cliente.get(
            "bienvenida_whatsapp_enviada", 0
        ),
    })
    # Extras
    for c in listar_contactos(cliente["id"]):
        out.append({
            "origen": "extra",
            "id": c["id"],
            "nombre": c.get("nombre", ""),
            "rol": c.get("rol", "") or "Contacto",
            "email": (c.get("email") or "").strip(),
            "whatsapp": (c.get("whatsapp") or "").strip(),
            "alertas_email_activas": c.get("alertas_email_activas", 1),
            "alertas_whatsapp_activas": c.get("alertas_whatsapp_activas", 1),
            "bienvenida_email_enviada": c.get("bienvenida_email_enviada", 0),
            "bienvenida_whatsapp_enviada": c.get(
                "bienvenida_whatsapp_enviada", 0
            ),
        })
    return out


def dias_alerta_consecutivos_previos(cliente_id: int,
                                        hasta_fecha: str) -> int:
    """Cuenta días consecutivos previos con alerta diaria enviada
    (n_alertas > 0). NO incluye el día actual.

    Sirve para detectar la etapa del evento:
      - 0 días previos + alerta hoy → INICIO
      - 1-2 días previos → PERSISTENCIA
      - 3+ días previos → ACUMULACIÓN
      - 1+ días previos + NORMAL hoy → RECUPERACIÓN
    """
    from datetime import datetime as _dt, timedelta as _td
    try:
        hoy = _dt.strptime(hasta_fecha, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0
    racha = 0
    with get_conn() as conn:
        for offset in range(1, 10):
            f = (hoy - _td(days=offset)).isoformat()
            r = conn.execute(
                "SELECT 1 FROM alertas_enviadas "
                "WHERE cliente_id = ? AND fecha = ? "
                "AND tipo = 'diaria' AND estado = 'enviada' "
                "AND n_alertas > 0 LIMIT 1",
                (cliente_id, f),
            ).fetchone()
            if r:
                racha += 1
            else:
                break
    return racha


def guardar_snapshot_pronostico(cliente_id: int, fecha_lunes: str,
                                  snapshot: list) -> None:
    """Guarda snapshot del pronóstico semanal (lunes).

    snapshot = [{fecha, severidad, tipo}, ...] — un item por día.
    Si ya existe un snapshot para ese cliente+lunes, lo reemplaza.
    """
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM pronosticos_semanales "
            "WHERE cliente_id = ? AND fecha_lunes = ?",
            (cliente_id, fecha_lunes),
        )
        conn.execute(
            "INSERT INTO pronosticos_semanales "
            "(cliente_id, fecha_lunes, snapshot_json) VALUES (?, ?, ?)",
            (cliente_id, fecha_lunes, json.dumps(snapshot)),
        )


def obtener_snapshot_pronostico(cliente_id: int,
                                  fecha_lunes: str) -> Optional[list]:
    """Devuelve el snapshot del lunes anterior para un cliente, o None."""
    with get_conn() as conn:
        r = conn.execute(
            "SELECT snapshot_json FROM pronosticos_semanales "
            "WHERE cliente_id = ? AND fecha_lunes = ? "
            "ORDER BY fecha_creacion DESC LIMIT 1",
            (cliente_id, fecha_lunes),
        ).fetchone()
        if not r:
            return None
        try:
            return json.loads(r["snapshot_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None


def marcar_bienvenida_enviada(origen: str, id_: int, canal: str) -> None:
    """Marca la flag de bienvenida como enviada.

    origen: 'principal' (tabla clientes) o 'extra' (tabla contactos)
    canal:  'email' o 'whatsapp'
    """
    if canal not in ("email", "whatsapp"):
        return
    columna = (f"bienvenida_{canal}_enviada")
    tabla = "clientes" if origen == "principal" else "contactos"
    with get_conn() as conn:
        conn.execute(
            f"UPDATE {tabla} SET {columna} = 1 WHERE id = ?", (id_,)
        )


# =====================================================================
# LOTES
# =====================================================================

def crear_lote(cliente_id: int, identificador: str, corral: str = "",
               raza: str = "", categoria: str = "",
               fecha_ingreso: str = "", cantidad_inicial: int = 0,
               peso_ingreso_kg: float = 0, objetivo_peso_kg: float = 0,
               objetivo_fecha: str = "", notas: str = "",
               adpv_objetivo_kg: Optional[float] = None,
               energia_dieta_mcal_em_kg_ms: Optional[float] = None) -> int:
    if not fecha_ingreso:
        fecha_ingreso = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO lotes (cliente_id, identificador, corral, raza, "
            "categoria, fecha_ingreso, cantidad_inicial, peso_ingreso_kg, "
            "objetivo_peso_kg, objetivo_fecha, notas, adpv_objetivo_kg, "
            "energia_dieta_mcal_em_kg_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cliente_id, identificador, corral, raza, categoria,
             fecha_ingreso, cantidad_inicial, peso_ingreso_kg,
             objetivo_peso_kg, objetivo_fecha, notas,
             adpv_objetivo_kg, energia_dieta_mcal_em_kg_ms),
        )
        return cur.lastrowid


def listar_lotes(cliente_id: Optional[int] = None,
                  estado: Optional[str] = None) -> List[Dict]:
    """Devuelve lotes con datos del cliente y la última pesada."""
    where = []
    params = []
    if cliente_id:
        where.append("l.cliente_id = ?")
        params.append(cliente_id)
    if estado:
        where.append("l.estado = ?")
        params.append(estado)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT l.*, c.nombre AS cliente_nombre, c.establecimiento,
               (SELECT COUNT(*) FROM pesadas WHERE lote_id = l.id) AS n_pesadas,
               (SELECT MAX(fecha) FROM pesadas WHERE lote_id = l.id) AS ultima_pesada,
               (SELECT peso_promedio_kg FROM pesadas WHERE lote_id = l.id
                ORDER BY fecha DESC LIMIT 1) AS ultimo_peso_kg
        FROM lotes l
        JOIN clientes c ON c.id = l.cliente_id
        {where_sql}
        ORDER BY l.fecha_creacion DESC
    """
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def obtener_lote(lote_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        r = conn.execute(
            """SELECT l.*, c.nombre AS cliente_nombre, c.establecimiento, c.contacto
               FROM lotes l JOIN clientes c ON c.id = l.cliente_id
               WHERE l.id = ?""", (lote_id,)
        ).fetchone()
        return dict(r) if r else None


def actualizar_lote(lote_id: int, **campos) -> None:
    if not campos:
        return
    sets = ", ".join(f"{k} = ?" for k in campos)
    valores = list(campos.values()) + [lote_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE lotes SET {sets} WHERE id = ?", valores)


def eliminar_lote(lote_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM lotes WHERE id = ?", (lote_id,))


# =====================================================================
# MOVIMIENTOS DE HACIENDA (cambios de cantidad en un lote)
# =====================================================================

# Tipos de movimiento y su signo sobre la cantidad del lote.
# Los que SUMAN animales al lote tienen signo +1; los que SACAN, -1.
MOVIMIENTO_TIPOS = {
    "muerte": -1,
    "venta": -1,
    "traslado_egreso": -1,
    "traslado_ingreso": +1,
    "ingreso": +1,
}

# Etiquetas amigables para mostrar al usuario.
MOVIMIENTO_LABELS = {
    "muerte": "Muerte / mortandad",
    "venta": "Venta (gordo / descarte)",
    "traslado_egreso": "Traslado (sale del lote)",
    "traslado_ingreso": "Traslado (entra al lote)",
    "ingreso": "Ingreso (compra / nacimiento)",
}


def crear_movimiento_lote(
    lote_id: int,
    fecha: str,
    tipo: str,
    cantidad: int,
    kg_promedio_animal: Optional[float] = None,
    destino_lote_id: Optional[int] = None,
    detalles: str = "",
) -> int:
    """Registra un movimiento de hacienda en un lote.

    Si `tipo` es 'traslado_egreso' y se pasa `destino_lote_id`,
    automáticamente se crea también el movimiento 'traslado_ingreso'
    en el lote destino (ambos en la misma transacción).

    Args:
        lote_id: id del lote origen del movimiento.
        fecha: 'YYYY-MM-DD'. Fecha en que ocurrió.
        tipo: uno de MOVIMIENTO_TIPOS.
        cantidad: cantidad de animales (siempre positiva; el signo lo
            define el tipo).
        kg_promedio_animal: peso por animal al momento del movimiento.
            Útil en ventas (kg vendidos) y muertes (peso del animal
            perdido).
        destino_lote_id: en traslados, lote al que va el animal.
        detalles: observaciones libres.

    Returns:
        id del movimiento creado (el del egreso si hubo traslado).
    """
    if tipo not in MOVIMIENTO_TIPOS:
        raise ValueError(
            f"Tipo de movimiento inválido: {tipo}. "
            f"Válidos: {list(MOVIMIENTO_TIPOS.keys())}"
        )
    if cantidad <= 0:
        raise ValueError("La cantidad debe ser un entero positivo.")

    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO movimientos
               (lote_id, fecha, tipo, cantidad, kg_promedio_animal,
                destino_lote_id, detalles)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (lote_id, fecha, tipo, int(cantidad),
             kg_promedio_animal, destino_lote_id, detalles),
        )
        mov_id = cur.lastrowid

        # Si es un traslado_egreso con destino conocido, generar el
        # traslado_ingreso en el lote destino.
        if tipo == "traslado_egreso" and destino_lote_id:
            conn.execute(
                """INSERT INTO movimientos
                   (lote_id, fecha, tipo, cantidad,
                    kg_promedio_animal, destino_lote_id, detalles)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (destino_lote_id, fecha, "traslado_ingreso",
                 int(cantidad), kg_promedio_animal, lote_id,
                 detalles or f"Traslado desde lote #{lote_id}"),
            )
        return mov_id


def listar_movimientos_lote(
    lote_id: int, hasta_fecha: Optional[str] = None,
) -> List[Dict]:
    """Devuelve los movimientos de un lote ordenados por fecha asc."""
    q = ("SELECT * FROM movimientos WHERE lote_id = ? "
         "AND (? IS NULL OR fecha <= ?) "
         "ORDER BY fecha ASC, id ASC")
    with get_conn() as conn:
        rows = conn.execute(
            q, (lote_id, hasta_fecha, hasta_fecha)
        ).fetchall()
        return [dict(r) for r in rows]


def cantidad_vigente_lote(
    lote_id: int, fecha: Optional[str] = None,
) -> int:
    """Cantidad de animales vigente en el lote a una fecha dada.

    Fórmula:
        cantidad_inicial + Σ(movimientos hasta `fecha`)

    Si `fecha` es None, devuelve la cantidad actual (hoy en adelante).

    Returns:
        int con la cantidad vigente. Nunca menor a 0.
    """
    with get_conn() as conn:
        lote = conn.execute(
            "SELECT cantidad_inicial FROM lotes WHERE id = ?",
            (lote_id,),
        ).fetchone()
        if not lote:
            return 0
        base = int(lote["cantidad_inicial"] or 0)

        q = ("SELECT tipo, cantidad FROM movimientos "
             "WHERE lote_id = ? "
             "AND (? IS NULL OR fecha <= ?)")
        rows = conn.execute(q, (lote_id, fecha, fecha)).fetchall()

    delta = 0
    for r in rows:
        signo = MOVIMIENTO_TIPOS.get(r["tipo"], 0)
        delta += signo * int(r["cantidad"] or 0)

    return max(0, base + delta)


def eliminar_movimiento_lote(mov_id: int) -> None:
    """Elimina un movimiento. Si era parte de un traslado, también
    elimina el movimiento espejo en el lote destino (mismo fecha,
    cantidad y tipo opuesto)."""
    with get_conn() as conn:
        mov = conn.execute(
            "SELECT * FROM movimientos WHERE id = ?", (mov_id,)
        ).fetchone()
        if not mov:
            return
        conn.execute("DELETE FROM movimientos WHERE id = ?", (mov_id,))
        # Borrar espejo si era traslado
        if mov["tipo"] == "traslado_egreso" and mov["destino_lote_id"]:
            conn.execute(
                """DELETE FROM movimientos
                   WHERE lote_id = ? AND tipo = 'traslado_ingreso'
                   AND fecha = ? AND cantidad = ?
                   AND destino_lote_id = ?""",
                (mov["destino_lote_id"], mov["fecha"],
                 mov["cantidad"], mov["lote_id"]),
            )
        elif mov["tipo"] == "traslado_ingreso" and mov["destino_lote_id"]:
            conn.execute(
                """DELETE FROM movimientos
                   WHERE lote_id = ? AND tipo = 'traslado_egreso'
                   AND fecha = ? AND cantidad = ?
                   AND destino_lote_id = ?""",
                (mov["destino_lote_id"], mov["fecha"],
                 mov["cantidad"], mov["lote_id"]),
            )


# =====================================================================
# CARGAS DEL SILOCOMEDERO
# =====================================================================

def crear_carga_silocomedero(
    lote_id: int, fecha_carga: str, kg_cargados: float,
    detalles: str = "",
    tipo_carga: str = "silo_carga",
    desglose_ingredientes: Optional[List[Dict]] = None,
    dias_cubiertos: float = 1.0,
    hora_carga: Optional[str] = None,
) -> int:
    """Registra una carga de mezcla en el comedero de un lote.

    Sirve para silocomedero (carga cada 3-7 días) y para comedero
    lineal (carga diaria, o varias comidas por día). El campo
    `tipo_carga` distingue ambos. Para el modo flexible (lineal con
    varias comidas por día), simplemente se registran varias entradas
    con la misma `fecha_carga` y distinta `hora_carga`.

    Args:
        lote_id: id del lote.
        fecha_carga: 'YYYY-MM-DD' del día que se cargó.
        kg_cargados: kg totales de mezcla volcados al comedero.
        detalles: observaciones libres.
        tipo_carga: 'silo_carga' (varios días) o 'lineal_diario' (1 día).
        desglose_ingredientes: lista opcional con
            [{"nombre": str, "kg": float}, ...] si el productor desglosó
            por ingrediente. Si está None, se asume sólo el total.
        dias_cubiertos: cuántos días dura esa carga (1 para lineal).
        hora_carga: 'HH:MM' opcional. Si no se pasa, queda NULL (los
            registros viejos no la tienen).

    Returns:
        id de la carga creada.
    """
    if kg_cargados <= 0:
        raise ValueError("kg_cargados debe ser positivo.")
    if dias_cubiertos <= 0:
        dias_cubiertos = 1.0
    desg_json = (
        json.dumps(desglose_ingredientes, ensure_ascii=False)
        if desglose_ingredientes else None
    )
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO cargas_silocomedero
               (lote_id, fecha_carga, kg_cargados, detalles,
                tipo_carga, desglose_ingredientes_json,
                dias_cubiertos, hora_carga)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (lote_id, fecha_carga, float(kg_cargados), detalles,
             tipo_carga, desg_json, float(dias_cubiertos),
             hora_carga),
        )
        return cur.lastrowid


def listar_cargas_silocomedero(
    lote_id: int, limit: int = 30,
) -> List[Dict]:
    """Cargas de un lote ordenadas de más nueva a más vieja.

    Cada dict incluye `desglose_ingredientes` (lista) parseada del JSON
    si está cargada, sino [].
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM cargas_silocomedero
               WHERE lote_id = ?
               ORDER BY fecha_carga DESC, id DESC LIMIT ?""",
            (lote_id, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["desglose_ingredientes"] = json.loads(
                    d.get("desglose_ingredientes_json") or "[]"
                )
            except Exception:
                d["desglose_ingredientes"] = []
            out.append(d)
        return out


def ultima_carga_silocomedero(lote_id: int) -> Optional[Dict]:
    """Devuelve la carga más reciente del SILO del lote, o None si no
    hay.

    IMPORTANTE: excluye cargas de `rollo_libre` (rollo a discreción).
    Esas son entregas de forraje separadas y no representan la carga
    del silocomedero — si se incluyeran, los cálculos de autonomía y
    "días por carga" del producto HMS se romperían (el rollo no tiene
    el ingrediente HMS adentro).
    """
    with get_conn() as conn:
        r = conn.execute(
            """SELECT * FROM cargas_silocomedero
               WHERE lote_id = ?
                 AND (tipo_carga IS NULL
                      OR LOWER(tipo_carga) != 'rollo_libre')
               ORDER BY fecha_carga DESC, id DESC LIMIT 1""",
            (lote_id,),
        ).fetchone()
        return dict(r) if r else None


def eliminar_carga_silocomedero(carga_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM cargas_silocomedero WHERE id = ?", (carga_id,)
        )


def actualizar_carga_silocomedero(
    carga_id: int,
    fecha_carga: Optional[str] = None,
    kg_cargados: Optional[float] = None,
    detalles: Optional[str] = None,
    tipo_carga: Optional[str] = None,
    desglose_ingredientes: Optional[List[Dict]] = None,
    dias_cubiertos: Optional[float] = None,
    hora_carga: Optional[str] = None,
) -> None:
    """Actualiza una carga ya registrada. Solo modifica los campos que
    se pasan (los demás quedan como estaban).

    Útil para corregir entregas de rollo cuyo sistema de oferta o
    cantidad fue cargada con un dato incorrecto, sin necesidad de
    borrar y rehacer.
    """
    campos = []
    valores: List = []
    if fecha_carga is not None:
        campos.append("fecha_carga = ?")
        valores.append(fecha_carga)
    if kg_cargados is not None:
        if kg_cargados <= 0:
            raise ValueError("kg_cargados debe ser positivo.")
        campos.append("kg_cargados = ?")
        valores.append(float(kg_cargados))
    if detalles is not None:
        campos.append("detalles = ?")
        valores.append(detalles)
    if tipo_carga is not None:
        campos.append("tipo_carga = ?")
        valores.append(tipo_carga)
    if desglose_ingredientes is not None:
        campos.append("desglose_ingredientes_json = ?")
        valores.append(
            json.dumps(desglose_ingredientes, ensure_ascii=False)
        )
    if dias_cubiertos is not None:
        if dias_cubiertos <= 0:
            dias_cubiertos = 1.0
        campos.append("dias_cubiertos = ?")
        valores.append(float(dias_cubiertos))
    if hora_carga is not None:
        campos.append("hora_carga = ?")
        valores.append(hora_carga)

    if not campos:
        return

    valores.append(carga_id)
    sql = (
        f"UPDATE cargas_silocomedero SET {', '.join(campos)} "
        f"WHERE id = ?"
    )
    with get_conn() as conn:
        conn.execute(sql, valores)


# =====================================================================
# DEDUP DE PEDIDOS DE CARGA POR WHATSAPP
# =====================================================================

def listar_avisos_enviados(
    cliente_id: Optional[int] = None,
    dias: int = 30,
    limit: int = 100,
) -> List[Dict]:
    """Avisos enviados (email + WhatsApp) cronológicamente.

    Unifica las dos tablas:
      - alertas_enviadas (email)
      - alertas_whatsapp_enviadas (whatsapp)

    Args:
        cliente_id: si se pasa, filtra por ese cliente. None = todos.
        dias: ventana hacia atrás en días.
        limit: máximo de filas devueltas.

    Returns:
        Lista de dicts ordenados por fecha desc, cada uno con:
          - canal: 'email' / 'whatsapp'
          - fecha_creacion
          - cliente_id, cliente_nombre
          - destinatario
          - asunto / mensaje (recortado)
          - tipo (para email: 'diaria', 'critica', 'demanda', etc.)
          - estado
          - error
    """
    fecha_desde = (
        datetime.now() - timedelta(days=int(dias))
    ).strftime("%Y-%m-%d %H:%M:%S")
    out: List[Dict] = []
    with get_conn() as conn:
        # EMAIL
        sql_email = """
            SELECT a.id, a.fecha_creacion, a.cliente_id,
                   a.destinatario, a.asunto, a.tipo, a.estado,
                   a.error, c.nombre AS cliente_nombre
            FROM alertas_enviadas a
            LEFT JOIN clientes c ON c.id = a.cliente_id
            WHERE a.fecha_creacion >= ?
        """
        params: List = [fecha_desde]
        if cliente_id is not None:
            sql_email += " AND a.cliente_id = ?"
            params.append(int(cliente_id))
        sql_email += " ORDER BY a.fecha_creacion DESC LIMIT ?"
        params.append(int(limit))
        for r in conn.execute(sql_email, params).fetchall():
            d = dict(r)
            d["canal"] = "email"
            d["mensaje"] = d.get("asunto") or ""
            out.append(d)

        # WHATSAPP
        sql_wa = """
            SELECT w.id, w.fecha_creacion, w.cliente_id,
                   w.destinatario, w.clave_dedup, w.estado,
                   w.error, w.mensaje, c.nombre AS cliente_nombre
            FROM alertas_whatsapp_enviadas w
            LEFT JOIN clientes c ON c.id = w.cliente_id
            WHERE w.fecha_creacion >= ?
        """
        params_w: List = [fecha_desde]
        if cliente_id is not None:
            sql_wa += " AND w.cliente_id = ?"
            params_w.append(int(cliente_id))
        sql_wa += " ORDER BY w.fecha_creacion DESC LIMIT ?"
        params_w.append(int(limit))
        for r in conn.execute(sql_wa, params_w).fetchall():
            d = dict(r)
            d["canal"] = "whatsapp"
            d["asunto"] = d.get("clave_dedup") or ""
            d["tipo"] = d.get("clave_dedup") or ""
            out.append(d)

    # Ordenar todo por fecha_creacion desc y recortar al límite global
    def _key(x: Dict) -> str:
        return x.get("fecha_creacion") or ""
    out.sort(key=_key, reverse=True)
    return out[:limit]


def pedido_carga_ya_enviado(
    lote_id: int, fecha: str, comida_n: int, intento_n: int = 1,
) -> bool:
    """¿Ya se mandó WhatsApp por este (lote, fecha, comida_n, intento_n)?"""
    with get_conn() as conn:
        r = conn.execute(
            """SELECT id FROM pedidos_carga_enviados
               WHERE lote_id = ? AND fecha = ?
                 AND comida_n = ? AND intento_n = ?
               LIMIT 1""",
            (lote_id, fecha, comida_n, intento_n),
        ).fetchone()
        return r is not None


def registrar_pedido_carga_enviado(
    lote_id: int, fecha: str, comida_n: int, intento_n: int = 1,
) -> None:
    """Marca que ya se mandó WhatsApp para esa comida del día."""
    with get_conn() as conn:
        try:
            conn.execute(
                """INSERT INTO pedidos_carga_enviados
                   (lote_id, fecha, comida_n, intento_n)
                   VALUES (?, ?, ?, ?)""",
                (lote_id, fecha, comida_n, intento_n),
            )
        except Exception:
            # UNIQUE constraint — ya estaba.
            pass


def calcular_fecha_objetivo_estimada(
    fecha_ingreso: str,
    peso_ingreso_kg: float,
    peso_objetivo_kg: float,
    adpv_kg_dia: Optional[float] = None,
    categoria: str = "",
) -> Optional[Dict]:
    """Calcula los días de encierre y la fecha objetivo de salida de
    un lote en base a la ganancia diaria esperada.

    Fórmula:
        días_encierre = (peso_objetivo - peso_ingreso) / ADPV
        fecha_objetivo = fecha_ingreso + días_encierre

    Si no se pasa `adpv_kg_dia` o es 0, usa el default por categoría
    (impacto_productivo.DEFAULTS_CATEGORIA).

    Args:
        fecha_ingreso: ISO YYYY-MM-DD.
        peso_ingreso_kg: peso al ingreso (kg).
        peso_objetivo_kg: peso objetivo de salida (kg).
        adpv_kg_dia: ganancia diaria esperada (kg/día). Opcional.
        categoria: categoría del lote para inferir ADPV si no se pasa.

    Returns:
        Dict con dias_encierre (int), fecha_objetivo (ISO),
        adpv_usado (float), fuente_adpv ("ingresado" | "default
        categoría"). None si faltan datos críticos.
    """
    from datetime import datetime as _dt, timedelta as _td

    if peso_ingreso_kg <= 0 or peso_objetivo_kg <= 0:
        return None
    if peso_objetivo_kg <= peso_ingreso_kg:
        return None  # no tiene sentido si el objetivo es bajar peso
    if not fecha_ingreso:
        return None

    # Resolver ADPV
    adpv = float(adpv_kg_dia or 0)
    fuente = "ingresado"
    if adpv <= 0:
        try:
            from .impacto_productivo import (
                DEFAULTS_CATEGORIA, _normalizar_categoria,
            )
            cat_norm = _normalizar_categoria(categoria)
            defaults = DEFAULTS_CATEGORIA.get(cat_norm) or {}
            adpv = float(defaults.get("adpv_objetivo_kg") or 0)
            fuente = f"default categoría '{cat_norm}'"
        except Exception:
            adpv = 0
    if adpv <= 0:
        return None

    try:
        d_ing = _dt.strptime(fecha_ingreso[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

    dias = max(1, int(round(
        (peso_objetivo_kg - peso_ingreso_kg) / adpv
    )))
    fecha_obj = (d_ing + _td(days=dias)).isoformat()

    return {
        "dias_encierre": dias,
        "fecha_objetivo": fecha_obj,
        "adpv_usado": round(adpv, 3),
        "fuente_adpv": fuente,
    }


def diagnostico_alimentacion_lote(
    lote: Dict, dias_adaptacion: int = 15,
) -> Dict:
    """Devuelve un diagnóstico operativo del sistema de entrega del lote
    — qué tan rápido se puede ajustar la ración ante un evento climático.

    Lógica:
      • Si está dentro de los primeros `dias_adaptacion` días desde
        `fecha_ingreso`, asume comedero LINEAL DIARIO sin importar lo
        que esté cargado (regla del campo: en adaptación siempre se
        controla diario).
      • Después, usa los campos `tipo_comedero_concentrado`,
        `forraje_modalidad` y `frecuencia_mezcla_dias` cargados en la
        ficha.

    Args:
        lote: dict con los campos del lote (de obtener_lote).
        dias_adaptacion: días desde el ingreso que se consideran
            adaptación obligatoria. Default 15.

    Returns:
        Dict con:
        - 'en_adaptacion': bool, True si está en los primeros N días
        - 'dias_desde_ingreso': int
        - 'tipo_comedero_efectivo': el comedero efectivo (forzado a
            lineal si está en adaptación)
        - 'frecuencia_efectiva_dias': cada cuántos días se puede ajustar
            (1 si adaptación o lineal diario; sino el del lote)
        - 'forraje_modalidad': mezclado / aparte / desconocido
        - 'puede_cambiar_mezcla_inmediato': bool — True si la próxima
            preparación es hoy/mañana; False si hay inercia mayor
        - 'descripcion': frase corta para mostrar en UI/log
    """
    from datetime import datetime as _dt
    fecha_ingreso_s = lote.get("fecha_ingreso") or ""
    dias_desde = None
    if fecha_ingreso_s:
        try:
            fi = _dt.strptime(fecha_ingreso_s[:10], "%Y-%m-%d").date()
            dias_desde = (_dt.now().date() - fi).days
        except (ValueError, TypeError):
            dias_desde = None

    en_adaptacion = (
        dias_desde is not None and 0 <= dias_desde <= dias_adaptacion
    )

    tipo_cargado = (lote.get("tipo_comedero_concentrado") or "").strip()
    forraje = (lote.get("forraje_modalidad") or "").strip()
    frec_cargada = lote.get("frecuencia_mezcla_dias") or 0
    try:
        frec_cargada = int(frec_cargada)
    except (TypeError, ValueError):
        frec_cargada = 0

    if en_adaptacion:
        tipo_efectivo = "lineal"
        frec_efectiva = 1
        descripcion = (
            f"En adaptación (día {dias_desde} de {dias_adaptacion}) — "
            f"comedero lineal diario obligatorio, ajustes pueden "
            f"hacerse al día siguiente."
        )
    else:
        tipo_efectivo = tipo_cargado or "desconocido"
        # Para lineal asumimos diario si no hay duración cargada (es
        # lo más común). Para silocomedero / autoconsumo no asumimos
        # nada — la duración la tiene que cargar el asesor según el
        # caso real, varía mucho por tamaño del silo y consumo.
        if tipo_efectivo == "lineal" and not frec_cargada:
            frec_efectiva = 1
        else:
            frec_efectiva = frec_cargada or 0

        if tipo_efectivo == "desconocido":
            descripcion = (
                "Sin tipo de comedero cargado — no se sabe la inercia "
                "del sistema. Cargá tipo de comedero y duración de "
                "carga en la ficha para que las alertas se adapten."
            )
        elif tipo_efectivo in ("silocomedero", "autoconsumo") and not frec_cargada:
            descripcion = (
                f"{tipo_efectivo.capitalize()} — falta cargar la "
                f"duración típica de una carga (varía por tamaño del "
                f"silo y consumo del lote). Mientras tanto las "
                f"alertas usarán lógica conservadora."
            )
        else:
            modal_str = ""
            if forraje == "mezclado":
                modal_str = " · forraje mezclado con la ración"
            elif forraje == "aparte":
                modal_str = " · forraje aparte (autoconsumo o corral)"
            if frec_efectiva == 1:
                tiempo_str = "carga diaria"
            else:
                tiempo_str = f"cada carga dura {frec_efectiva} día(s)"
            descripcion = (
                f"{tipo_efectivo.capitalize()} · {tiempo_str}{modal_str}."
            )

    # Solo se considera "puede ajustar inmediato" si el sistema está
    # bien definido Y la frecuencia es 1 día (lineal diario o en
    # adaptación). Sin info, asumimos lo conservador: hay inercia.
    if tipo_efectivo == "desconocido" or not frec_efectiva:
        puede_inmediato = False
    else:
        puede_inmediato = frec_efectiva <= 1

    return {
        "en_adaptacion": en_adaptacion,
        "dias_desde_ingreso": dias_desde,
        "tipo_comedero_efectivo": tipo_efectivo,
        "frecuencia_efectiva_dias": frec_efectiva,
        "forraje_modalidad": forraje or "desconocido",
        "puede_cambiar_mezcla_inmediato": puede_inmediato,
        "descripcion": descripcion,
    }


# =====================================================================
# HISTÓRICO DE IMPACTOS PRODUCTIVOS POR LOTE
# =====================================================================

def guardar_impacto_lote(lote_id: int, impacto: Dict,
                          tipo_evento: str = "frio",
                          severidad: str = "operativo",
                          fecha_inicio_evento: Optional[str] = None,
                          fecha_fin_evento: Optional[str] = None,
                          clima_resumen: Optional[Dict] = None,
                          notas: str = "",
                          estado: str = "proyectado",
                          dedup_semana: bool = True) -> Optional[int]:
    """Guarda un cálculo de impacto productivo en el histórico.

    Args:
        lote_id: ID del lote.
        impacto: dict devuelto por estimar_impacto_frio (rango y supuestos).
        tipo_evento: "frio" o "calor".
        severidad: "atencion" | "operativo" | "critico".
        fecha_inicio_evento: ISO date string del inicio del evento.
        fecha_fin_evento: ISO date string del fin del evento.
        clima_resumen: dict con T° min/max, viento, HR, lluvia, barro.
        notas: texto libre.
        estado: "proyectado" (cálculo con pronóstico futuro) o
            "confirmado" (recalculado con clima histórico real).
        dedup_semana: si True, no guarda si ya hay un impacto del
            mismo `estado` para ese lote en la misma semana ISO.

    Returns:
        id del registro guardado o None si fue dedup.
    """
    if not impacto:
        return None
    fecha_calc = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    iso_year, iso_week, _ = datetime.now().isocalendar()
    semana_key = f"{iso_year}-W{iso_week:02d}"

    g_min, g_max = impacto.get("gasto_extra_pct", (None, None))
    a_min, a_max = impacto.get("adpv_perdida_kg_rango", (None, None))
    p_min, p_max = impacto.get("pct_adpv_perdida", (None, None))
    k_min, k_max = impacto.get("kg_perdidos_lote_periodo", (None, None))

    import json as _json
    clima_json = _json.dumps(clima_resumen) if clima_resumen else None

    with get_conn() as conn:
        if dedup_semana:
            # Buscar si ya hay un registro del MISMO estado para esta
            # semana (un "proyectado" y un "confirmado" pueden
            # coexistir para la misma semana).
            rows = conn.execute(
                "SELECT id, fecha_calculo, estado FROM impactos_lote "
                "WHERE lote_id = ? AND estado = ? "
                "ORDER BY fecha_calculo DESC LIMIT 5",
                (lote_id, estado),
            ).fetchall()
            for r in rows:
                try:
                    # Si es "proyectado" usamos fecha_calculo;
                    # si es "confirmado" usamos fecha_inicio_evento
                    # para que la semana coincida con el evento, no
                    # con cuándo se confirmó.
                    ref = (fecha_inicio_evento[:10]
                           if estado == "confirmado" and fecha_inicio_evento
                           else r["fecha_calculo"][:10])
                    fc = datetime.strptime(ref, "%Y-%m-%d")
                    y, w, _ = fc.isocalendar()
                    if f"{y}-W{w:02d}" == semana_key:
                        return None
                except (ValueError, TypeError):
                    continue
            # Para confirmado, evaluamos la semana del EVENTO, no la
            # semana actual; recomputamos semana_key.
            if estado == "confirmado" and fecha_inicio_evento:
                try:
                    f_evt = datetime.strptime(
                        fecha_inicio_evento[:10], "%Y-%m-%d"
                    )
                    y, w, _ = f_evt.isocalendar()
                    semana_key_evt = f"{y}-W{w:02d}"
                    # Re-verificar dedup con la semana del evento
                    for r in rows:
                        try:
                            f_ie = r["fecha_calculo"][:10]
                            # buscar fecha_inicio del registro
                            r2 = conn.execute(
                                "SELECT fecha_inicio_evento FROM "
                                "impactos_lote WHERE id = ?",
                                (r["id"],),
                            ).fetchone()
                            if r2 and r2["fecha_inicio_evento"]:
                                fie = datetime.strptime(
                                    r2["fecha_inicio_evento"][:10],
                                    "%Y-%m-%d",
                                )
                                yy, ww, _ = fie.isocalendar()
                                if f"{yy}-W{ww:02d}" == semana_key_evt:
                                    return None
                        except (ValueError, TypeError):
                            continue
                except (ValueError, TypeError):
                    pass
        cur = conn.execute(
            """INSERT INTO impactos_lote
               (lote_id, fecha_calculo, fecha_inicio_evento,
                fecha_fin_evento, tipo_evento, severidad,
                gasto_extra_pct_min, gasto_extra_pct_max,
                adpv_perdida_min_kg, adpv_perdida_max_kg,
                kg_lote_total_min, kg_lote_total_max,
                pct_adpv_min, pct_adpv_max,
                dias_evento, cantidad_animales, peso_promedio_kg,
                clima_resumen_json, notas, estado)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (lote_id, fecha_calc, fecha_inicio_evento, fecha_fin_evento,
             tipo_evento, severidad, g_min, g_max, a_min, a_max,
             k_min, k_max, p_min, p_max,
             impacto.get("dias_evento"),
             impacto.get("cantidad_lote"),
             None, clima_json, notas, estado),
        )
        return cur.lastrowid


def listar_impactos_lote(lote_id: int,
                           desde: Optional[str] = None,
                           hasta: Optional[str] = None,
                           limit: int = 50) -> List[Dict]:
    """Lista los impactos guardados de un lote ordenados por fecha
    descendente. Opcionalmente filtra por rango de fechas."""
    where = ["lote_id = ?"]
    params: List = [lote_id]
    if desde:
        where.append("fecha_calculo >= ?")
        params.append(desde)
    if hasta:
        where.append("fecha_calculo <= ?")
        params.append(hasta)
    where_sql = " AND ".join(where)
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM impactos_lote WHERE {where_sql} "
            f"ORDER BY fecha_calculo DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]


def eliminar_impacto_lote(impacto_id: int) -> None:
    """Borra un registro de impacto del histórico."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM impactos_lote WHERE id = ?", (impacto_id,),
        )


# =====================================================================
# PESADAS
# =====================================================================

def guardar_pesada(lote_id: int, fecha: str, metodo: str,
                    cantidad_animales: int, peso_promedio_kg: float,
                    peso_total_kg: float, desvio_kg: float,
                    pesos_individuales: List[float] = None,
                    video_path: str = "", notas: str = "") -> int:
    cv = (desvio_kg / peso_promedio_kg * 100) if peso_promedio_kg else 0
    pesos_json = json.dumps(pesos_individuales or [])
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO pesadas (lote_id, fecha, metodo, cantidad_animales,
               peso_promedio_kg, peso_total_kg, desvio_kg, cv_pct,
               pesos_individuales_json, video_path, notas)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (lote_id, fecha, metodo, cantidad_animales,
             peso_promedio_kg, peso_total_kg, desvio_kg, cv,
             pesos_json, video_path, notas),
        )
        return cur.lastrowid


def listar_pesadas(lote_id: int) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM pesadas WHERE lote_id = ? ORDER BY fecha ASC",
            (lote_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["pesos_individuales"] = json.loads(d.get("pesos_individuales_json") or "[]")
            except json.JSONDecodeError:
                d["pesos_individuales"] = []
            result.append(d)
        return result


def eliminar_pesada(pesada_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM pesadas WHERE id = ?", (pesada_id,))


# =====================================================================
# DIETAS
# =====================================================================

def guardar_dieta(lote_id: int, fecha: str, composicion: List[Dict],
                   costo_dia: float = 0, pb_pct: float = 0,
                   em_mcal_dia: float = 0, consumo_ms_kg: float = 0,
                   nnp_pct: float = 0, observaciones: str = "") -> int:
    composicion_json = json.dumps(composicion or [])
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO dietas (lote_id, fecha, composicion_json, costo_dia,
               pb_pct, em_mcal_dia, consumo_ms_kg, nnp_pct, observaciones)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (lote_id, fecha, composicion_json, costo_dia, pb_pct,
             em_mcal_dia, consumo_ms_kg, nnp_pct, observaciones),
        )
        return cur.lastrowid


def listar_dietas(lote_id: int) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM dietas WHERE lote_id = ? ORDER BY fecha DESC",
            (lote_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["composicion"] = json.loads(d.get("composicion_json") or "[]")
            except json.JSONDecodeError:
                d["composicion"] = []
            result.append(d)
        return result


# =====================================================================
# ENTREGAS DE PRODUCTO (concentrados/núcleos al cliente)
# =====================================================================

def crear_entrega(
    cliente_id: int, producto_nombre: str, kg_total: float,
    fecha_entrega: str,
    lote_id: Optional[int] = None,
    formato: str = "granel",
    cantidad_bolsas: float = 0,
    kg_por_bolsa: float = 30,
    precio_kg: float = 0,
    precio_total: float = 0,
    notas: str = "",
) -> int:
    """Registra una entrega de producto a un cliente.

    Args:
        cliente_id: id del cliente que recibe.
        producto_nombre: marca específica (Fibrogreen, Fibroter, etc.)
        kg_total: kg de producto entregados (calculado o ingresado).
        fecha_entrega: ISO date (YYYY-MM-DD).
        lote_id: lote específico si la entrega está asociada a uno
            (opcional — si vale para todo el cliente, dejar None).
        formato: "granel" o "bolsa".
        cantidad_bolsas: si formato="bolsa", cuántas. Default 0.
        kg_por_bolsa: default 30 kg.
        precio_kg, precio_total: opcional.
        notas: texto libre.

    Returns:
        id de la entrega creada.
    """
    if not fecha_entrega:
        fecha_entrega = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO entregas_producto
               (cliente_id, lote_id, producto_nombre, formato,
                cantidad_bolsas, kg_por_bolsa, kg_total, fecha_entrega,
                precio_kg, precio_total, notas)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cliente_id, lote_id, producto_nombre, formato,
             cantidad_bolsas, kg_por_bolsa, kg_total, fecha_entrega,
             precio_kg, precio_total, notas),
        )
        return cur.lastrowid


def listar_entregas_cliente(cliente_id: int,
                              limit: int = 100) -> List[Dict]:
    """Devuelve todas las entregas de un cliente, ordenadas por fecha
    descendente."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.*, l.identificador AS lote_identificador
               FROM entregas_producto e
               LEFT JOIN lotes l ON l.id = e.lote_id
               WHERE e.cliente_id = ?
               ORDER BY e.fecha_entrega DESC LIMIT ?""",
            (cliente_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def listar_entregas_lote(lote_id: int) -> List[Dict]:
    """Devuelve entregas asociadas a un lote específico."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM entregas_producto
               WHERE lote_id = ?
               ORDER BY fecha_entrega DESC""",
            (lote_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def listar_entregas_periodo(
    fecha_desde: str, fecha_hasta: str,
) -> List[Dict]:
    """Devuelve todas las entregas (de cualquier cliente) hechas en el
    rango de fechas. Útil para KPIs del dashboard: kg entregados,
    facturado, productos top, etc.

    Args:
        fecha_desde: ISO YYYY-MM-DD inclusive.
        fecha_hasta: ISO YYYY-MM-DD inclusive.

    Returns:
        Lista de entregas con cliente_nombre y producto_nombre.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.*, c.nombre AS cliente_nombre,
                       l.identificador AS lote_identificador
               FROM entregas_producto e
               LEFT JOIN clientes c ON c.id = e.cliente_id
               LEFT JOIN lotes l ON l.id = e.lote_id
               WHERE e.fecha_entrega >= ? AND e.fecha_entrega <= ?
               ORDER BY e.fecha_entrega DESC""",
            (fecha_desde, fecha_hasta),
        ).fetchall()
        return [dict(r) for r in rows]


def eliminar_entrega(entrega_id: int) -> None:
    """Borra un registro de entrega."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM entregas_producto WHERE id = ?",
            (entrega_id,),
        )


def obtener_entrega(entrega_id: int) -> Optional[Dict]:
    """Devuelve una entrega por id."""
    with get_conn() as conn:
        r = conn.execute(
            "SELECT * FROM entregas_producto WHERE id = ?",
            (entrega_id,),
        ).fetchone()
        return dict(r) if r else None


def actualizar_entrega(entrega_id: int, **campos) -> None:
    """Edita una entrega existente. Soporta cualquier campo de la
    tabla `entregas_producto`. Útil para corregir precios cargados
    mal (ej. el clásico precio-por-bolsa-en-vez-de-precio-por-kg)
    sin tener que borrar y recrear la entrega."""
    if not campos:
        return
    sets = ", ".join(f"{k} = ?" for k in campos)
    valores = list(campos.values()) + [entrega_id]
    with get_conn() as conn:
        conn.execute(
            f"UPDATE entregas_producto SET {sets} WHERE id = ?",
            valores,
        )


# =====================================================================
# MÉTRICAS DEL LOTE
# =====================================================================

def calcular_evolucion_lote(lote_id: int) -> Dict:
    """Calcula ADG, ganancia total, días, etc. en base al histórico de pesadas."""
    pesadas = listar_pesadas(lote_id)
    if len(pesadas) < 2:
        return {
            "n_pesadas": len(pesadas),
            "primera_pesada": pesadas[0] if pesadas else None,
            "ultima_pesada": pesadas[0] if pesadas else None,
            "adg_total": 0,
            "ganancia_total_kg": 0,
            "dias_totales": 0,
            "tendencia": [],
        }

    primera = pesadas[0]
    ultima = pesadas[-1]
    f1 = datetime.strptime(primera["fecha"], "%Y-%m-%d")
    f2 = datetime.strptime(ultima["fecha"], "%Y-%m-%d")
    dias = max(1, (f2 - f1).days)
    ganancia = ultima["peso_promedio_kg"] - primera["peso_promedio_kg"]
    adg = ganancia / dias

    # Tendencia: ADG entre pesadas consecutivas
    tendencia = []
    for i in range(1, len(pesadas)):
        a, b = pesadas[i - 1], pesadas[i]
        f_a = datetime.strptime(a["fecha"], "%Y-%m-%d")
        f_b = datetime.strptime(b["fecha"], "%Y-%m-%d")
        d = max(1, (f_b - f_a).days)
        gan = b["peso_promedio_kg"] - a["peso_promedio_kg"]
        tendencia.append({
            "desde": a["fecha"], "hasta": b["fecha"], "dias": d,
            "ganancia_kg": gan, "adg": gan / d,
        })

    return {
        "n_pesadas": len(pesadas),
        "primera_pesada": primera,
        "ultima_pesada": ultima,
        "adg_total": adg,
        "ganancia_total_kg": ganancia,
        "dias_totales": dias,
        "tendencia": tendencia,
    }


def resumen_lote_para_ia(lote_id: int) -> str:
    """Genera un texto con todo el histórico del lote para inyectar en el agente IA."""
    lote = obtener_lote(lote_id)
    if not lote:
        return ""
    pesadas = listar_pesadas(lote_id)
    dietas = listar_dietas(lote_id)
    evol = calcular_evolucion_lote(lote_id)

    lines = [
        f"=== HISTÓRICO DEL LOTE ===",
        f"lote_id (DB): {lote_id}  ← USAR ESTE NÚMERO en cualquier "
        f"tool que pida 'lote_id' (guardar_dieta_lote, "
        f"calcular_dmi_proyectado_lote, formular_dieta_ajustada_por_clima)",
        f"Cliente: {lote['cliente_nombre']} ({lote.get('establecimiento','')})",
        f"Lote: {lote['identificador']} | Corral: {lote.get('corral') or '—'}",
        f"Raza/Categoría: {lote.get('raza','')} / {lote.get('categoria','')}",
        f"Cantidad inicial: {lote.get('cantidad_inicial', 0)} animales",
        f"Ingreso: {lote.get('fecha_ingreso','')} ({lote.get('peso_ingreso_kg', 0):.0f} kg promedio)",
    ]
    if lote.get("objetivo_peso_kg"):
        lines.append(
            f"Objetivo: {lote['objetivo_peso_kg']:.0f} kg para {lote.get('objetivo_fecha','—')}"
        )

    if pesadas:
        lines.append(f"\nPesadas registradas ({len(pesadas)}):")
        for p in pesadas:
            lines.append(
                f"  • {p['fecha']}: {p['cantidad_animales']} animales, "
                f"{p['peso_promedio_kg']:.1f} kg prom (CV {p.get('cv_pct',0):.1f}%) — "
                f"método {p.get('metodo','—')}"
            )
        if evol["adg_total"]:
            lines.append(
                f"\nEvolución global: +{evol['ganancia_total_kg']:.1f} kg en "
                f"{evol['dias_totales']} días → ADG {evol['adg_total']:.3f} kg/día"
            )

    if dietas:
        lines.append(f"\nDietas registradas ({len(dietas)}):")
        for d in dietas[:3]:  # las 3 últimas
            ings = ", ".join(f"{c.get('nombre','?')} {c.get('pct_ms',0):.0f}%"
                              for c in d.get("composicion", [])[:5])
            lines.append(
                f"  • {d['fecha']}: PB {d.get('pb_pct',0):.1f}%, "
                f"${d.get('costo_dia',0):.0f}/día — {ings}"
            )

    return "\n".join(lines)


# =====================================================================
# RECORDATORIOS DE LLAMADA AL CLIENTE
# Manual + automáticos. Vinculado por cliente_id.
# =====================================================================

def crear_recordatorio_llamada(
    cliente_id: int,
    fecha_objetivo: str,
    motivo: str = "",
    origen: str = "manual",
) -> int:
    """Crea un recordatorio de llamada pendiente. Devuelve el id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO recordatorios_llamada
               (cliente_id, fecha_objetivo, motivo, origen, estado)
               VALUES (?, ?, ?, ?, 'pendiente')""",
            (cliente_id, fecha_objetivo, motivo or "", origen),
        )
        return cur.lastrowid


def listar_recordatorios_pendientes(
    dias_hasta: int = 14,
    incluir_atrasados: bool = True,
) -> list:
    """Recordatorios pendientes en una ventana de N días.

    Args:
        dias_hasta: cuántos días hacia adelante mirar (default 14).
        incluir_atrasados: si True, también devuelve los que ya
            vencieron y siguen pendientes.

    Returns: lista de dicts con info del recordatorio + nombre cliente.
    """
    from datetime import datetime as _dt, timedelta as _td
    hoy = _dt.now().date()
    tope = (hoy + _td(days=dias_hasta)).isoformat()
    desde = "1900-01-01" if incluir_atrasados else hoy.isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT r.*, c.nombre AS cliente_nombre,
                      c.localidad AS cliente_localidad
               FROM recordatorios_llamada r
               JOIN clientes c ON c.id = r.cliente_id
               WHERE r.estado = 'pendiente'
                 AND date(r.fecha_objetivo) >= date(?)
                 AND date(r.fecha_objetivo) <= date(?)
               ORDER BY date(r.fecha_objetivo) ASC, r.id ASC""",
            (desde, tope),
        ).fetchall()
        return [dict(r) for r in rows]


def listar_recordatorios_cliente(
    cliente_id: int, incluir_completados: bool = True,
) -> list:
    """Todos los recordatorios de un cliente, más recientes primero."""
    with get_conn() as conn:
        if incluir_completados:
            sql = """SELECT * FROM recordatorios_llamada
                     WHERE cliente_id = ?
                     ORDER BY date(fecha_objetivo) DESC, id DESC"""
            rows = conn.execute(sql, (cliente_id,)).fetchall()
        else:
            sql = """SELECT * FROM recordatorios_llamada
                     WHERE cliente_id = ? AND estado = 'pendiente'
                     ORDER BY date(fecha_objetivo) ASC"""
            rows = conn.execute(sql, (cliente_id,)).fetchall()
        return [dict(r) for r in rows]


def marcar_recordatorio_hecho(
    rid: int,
    notas: str = "",
    evaluacion_json: str = "",
    lote_id: Optional[int] = None,
) -> bool:
    """Marca un recordatorio como completado.

    Args:
        rid: id del recordatorio.
        notas: markdown legible del cierre.
        evaluacion_json: JSON estructurado de las respuestas del
            cuestionario. Se usa para la ficha clínica del lote
            (agregados, patrones, tally de mortandad).
        lote_id: lote evaluado (si la conversación se enfocó en
            un lote específico). Se usa para vincular la
            evaluación a la ficha clínica del lote.
    """
    with get_conn() as conn:
        conn.execute(
            """UPDATE recordatorios_llamada
               SET estado = 'hecho',
                   notas_cierre = ?,
                   evaluacion_json = ?,
                   lote_id = COALESCE(?, lote_id),
                   completado_en = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (notas or "", evaluacion_json or "", lote_id, rid),
        )
        return True


def cancelar_recordatorio(rid: int) -> bool:
    """Marca un recordatorio como cancelado (no se borra para
    mantener trazabilidad)."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE recordatorios_llamada
               SET estado = 'cancelado',
                   completado_en = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (rid,),
        )
        return True


def reprogramar_recordatorio(rid: int, nueva_fecha: str) -> bool:
    """Cambia la fecha objetivo de un recordatorio pendiente."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE recordatorios_llamada
               SET fecha_objetivo = ?
               WHERE id = ? AND estado = 'pendiente'""",
            (nueva_fecha, rid),
        )
        return True


def _existe_recordatorio_auto(
    cliente_id: int, origen: str, ventana_dias: int = 14,
) -> bool:
    """¿Ya hay un recordatorio (pendiente o hecho) del mismo origen
    automático en la ventana? Sirve para evitar duplicados al
    generar sugerencias.
    """
    from datetime import datetime as _dt, timedelta as _td
    desde = (_dt.now().date() - _td(days=ventana_dias)).isoformat()
    with get_conn() as conn:
        r = conn.execute(
            """SELECT COUNT(*) AS n FROM recordatorios_llamada
               WHERE cliente_id = ? AND origen = ?
                 AND date(creado_en) >= date(?)""",
            (cliente_id, origen, desde),
        ).fetchone()
        return (r["n"] if r else 0) > 0


def armar_ficha_revision_cliente(cliente_id: int) -> dict:
    """Devuelve un snapshot técnico del cliente listo para una llamada.

    Junta toda la info relevante para que el asesor llame al cliente
    sabiendo cómo está: lotes activos, dieta vigente, stock, autonomía
    del silocomedero, último clima adverso, último contacto registrado.

    Returns:
        dict con secciones:
        - cliente: {nombre, localidad, contactos[]}
        - lotes: [{identificador, categoría, días, pv_hoy, adg_obj,
                   estado_carga_silo, dieta_vigente}]
        - alertas_recientes: últimas 5 alertas climáticas
        - puntos_chequeo: lista de strings sugeridos
        - ultimo_contacto: fecha del último llamado hecho
    """
    from datetime import datetime as _dt, timedelta as _td
    out: Dict[str, Any] = {
        "cliente": {}, "lotes": [], "alertas_recientes": [],
        "puntos_chequeo": [], "ultimo_contacto": None,
    }
    with get_conn() as conn:
        c = conn.execute(
            "SELECT * FROM clientes WHERE id = ?", (cliente_id,),
        ).fetchone()
        if not c:
            return out
        out["cliente"] = {
            "nombre": c["nombre"],
            "localidad": c["localidad"] or "",
            "telefono": (
                (c["whatsapp"] if "whatsapp" in c.keys() else "")
                or ""
            ),
        }
        # Contactos del cliente
        try:
            cts = conn.execute(
                "SELECT * FROM contactos WHERE cliente_id = ? "
                "AND COALESCE(activo,1) = 1 ORDER BY id",
                (cliente_id,),
            ).fetchall()
            out["cliente"]["contactos"] = [dict(x) for x in cts]
        except Exception:
            out["cliente"]["contactos"] = []

        # Lotes activos
        try:
            lotes = conn.execute(
                "SELECT * FROM lotes WHERE cliente_id = ? "
                "AND estado = 'activo' ORDER BY fecha_ingreso DESC",
                (cliente_id,),
            ).fetchall()
        except Exception:
            lotes = []

        hoy = _dt.now().date()
        for lt in lotes:
            ld = dict(lt)
            f_ing = ld.get("fecha_ingreso", "")
            try:
                _fi = _dt.strptime(f_ing[:10], "%Y-%m-%d").date()
                dias = (hoy - _fi).days
            except Exception:
                dias = 0
            # PV proyectado HOY (estimación simple)
            pv_ing = float(ld.get("peso_ingreso_kg") or 0)
            adg = float(ld.get("adpv_objetivo_kg") or 0)
            pv_hoy = pv_ing + (adg * max(0, dias))

            # Dieta vigente: la más reciente CON FECHA <= HOY
            # (no agarrar una fase futura del plan adaptación).
            # Si la vigente no tiene composición detallada, buscar
            # una anterior del mismo lote que sí la tenga — sirve
            # como fallback para mostrar la fórmula aunque la fase
            # vigente solo haya guardado KPIs globales.
            d_vig = None
            d_vig_comp_from = None  # de qué fecha viene la comp
            try:
                d_rows_raw = conn.execute(
                    "SELECT * FROM dietas WHERE lote_id = ? "
                    "AND date(fecha) <= date('now') "
                    "ORDER BY date(fecha) DESC",
                    (ld["id"],),
                ).fetchall()
                d_rows = [dict(r) for r in d_rows_raw]
                # Parsear composiciones
                import json as _json
                for d in d_rows:
                    try:
                        d["composicion"] = _json.loads(
                            d.get("composicion_json") or "[]"
                        )
                    except Exception:
                        d["composicion"] = []
                if d_rows:
                    d_vig = d_rows[0]
                    # Si vigente no tiene composición, buscar la
                    # más reciente con datos.
                    if not d_vig.get("composicion"):
                        for dr in d_rows[1:]:
                            if dr.get("composicion"):
                                # Tomamos la composición de esa
                                # pero mantenemos los KPIs de la
                                # dieta vigente.
                                d_vig["composicion"] = (
                                    dr["composicion"]
                                )
                                d_vig_comp_from = dr.get(
                                    "fecha", ""
                                )
                                break
            except Exception:
                d_vig = None

            # Composición de la dieta vigente (ingrediente por
            # ingrediente con kg t/c y % ración) — para mostrar
            # la fórmula completa en la ficha técnica.
            #
            # OJO: para planes de adaptación el agente IA a veces
            # solo guarda pct_ms (no kg_ms ni kg_tal_cual). Tenemos
            # que computar los kg desde el % y el DMI total del
            # lote, así la tabla muestra cantidades reales aunque
            # el guardado original haya sido parcial.
            comp_vig = []
            if d_vig:
                _comp_raw = d_vig.get("composicion") or []
                _dmi_dieta = float(
                    d_vig.get("consumo_ms_kg") or 0
                )
                # Si los kg_ms vienen cargados, los uso. Si no,
                # los calculo desde pct_ms × DMI.
                _tot_ms_raw = sum(
                    float(c.get("kg_ms") or 0)
                    for c in _comp_raw
                )
                _usar_calculo = (
                    _tot_ms_raw <= 0 and _dmi_dieta > 0
                )
                for c in _comp_raw:
                    _kg_ms_c = float(c.get("kg_ms") or 0)
                    _kg_tc_c = float(c.get("kg_tal_cual") or 0)
                    _pct_ms_c = float(c.get("pct_ms") or 0)
                    # Computar kg_ms si falta
                    if _kg_ms_c <= 0 and _usar_calculo:
                        _kg_ms_c = _pct_ms_c / 100 * _dmi_dieta
                    # Computar kg_tal_cual si falta
                    # Heurística: granos/núcleos ≈ 88% MS,
                    # forrajes/rollos ≈ 85% MS. Default 88%.
                    if _kg_tc_c <= 0 and _kg_ms_c > 0:
                        _nom_low = (
                            c.get("nombre", "") or ""
                        ).lower()
                        _ms_pct = (
                            0.85
                            if any(k in _nom_low for k in [
                                "rollo", "fardo", "henolaje",
                                "silaje", "alfalfa", "pastura",
                            ])
                            else 0.88
                        )
                        _kg_tc_c = _kg_ms_c / _ms_pct
                    # Recalcular total MS para % ración correcto
                    _tot_para_pct = (
                        sum(
                            (
                                float(x.get("kg_ms") or 0)
                                if not _usar_calculo
                                else (
                                    float(x.get("pct_ms") or 0)
                                    / 100 * _dmi_dieta
                                )
                            )
                            for x in _comp_raw
                        ) or 1.0
                    )
                    comp_vig.append({
                        "nombre": c.get("nombre", "?"),
                        "pct_ms": _pct_ms_c,
                        "pct_racion": (
                            _kg_ms_c / _tot_para_pct * 100
                            if _tot_para_pct else 0
                        ),
                        "kg_ms": round(_kg_ms_c, 2),
                        "kg_tal_cual": round(_kg_tc_c, 2),
                    })

            out["lotes"].append({
                "id": ld["id"],
                "identificador": ld.get("identificador", ""),
                "categoria": ld.get("categoria", ""),
                "raza": ld.get("raza", ""),
                "cantidad": ld.get("cantidad_inicial", 0),
                "dias": dias,
                "pv_ingreso_kg": pv_ing,
                "pv_hoy_kg": round(pv_hoy, 0),
                "adg_obj": adg,
                "tipo_comedero": (
                    ld.get("tipo_comedero_concentrado") or ""
                ),
                "objetivo_fecha": ld.get("objetivo_fecha", ""),
                "dieta_vigente": (
                    {
                        "fecha": d_vig.get("fecha", ""),
                        "pb_pct": d_vig.get("pb_pct", 0),
                        "em_mcal_dia": d_vig.get(
                            "em_mcal_dia", 0,
                        ),
                        "consumo_ms_kg": d_vig.get(
                            "consumo_ms_kg", 0,
                        ),
                        "nnp_pct": d_vig.get("nnp_pct", 0),
                        "observaciones": (
                            d_vig.get("observaciones", "") or ""
                        )[:200],
                        "composicion": comp_vig,
                        # Si la composición la traemos de una
                        # dieta anterior (porque la vigente está
                        # incompleta), guardamos la fecha de
                        # origen para mostrarlo al usuario.
                        "composicion_origen_fecha": (
                            d_vig_comp_from or ""
                        ),
                    } if d_vig else None
                ),
            })

        # Últimas alertas enviadas a este cliente (3)
        try:
            als = conn.execute(
                """SELECT fecha, tipo, asunto, estado
                   FROM alertas_enviadas
                   WHERE cliente_id = ?
                   ORDER BY date(fecha) DESC LIMIT 3""",
                (cliente_id,),
            ).fetchall()
            out["alertas_recientes"] = [dict(x) for x in als]
        except Exception:
            pass

        # Último contacto registrado (llamado hecho)
        try:
            uc = conn.execute(
                """SELECT fecha_objetivo, completado_en, notas_cierre
                   FROM recordatorios_llamada
                   WHERE cliente_id = ? AND estado = 'hecho'
                   ORDER BY date(completado_en) DESC LIMIT 1""",
                (cliente_id,),
            ).fetchone()
            if uc:
                out["ultimo_contacto"] = dict(uc)
        except Exception:
            pass

    # Armar puntos de chequeo sugeridos
    pts = []
    for lt in out["lotes"]:
        ident = lt["identificador"]
        if lt["dias"] < 14:
            pts.append(
                f"🐂 **{ident}** ({lt['dias']}d en sistema): "
                "¿cómo viene la adaptación al concentrado? "
                "¿consumen bien? ¿hay diarreas o tos?"
            )
        if (lt["tipo_comedero"] or "").lower() == "silocomedero":
            pts.append(
                f"🛢️ **{ident}**: ¿necesita carga próxima del "
                "silo? ¿queda mezcla suficiente?"
            )
        if lt["dieta_vigente"] and "adaptacion" in (
            lt["dieta_vigente"].get("observaciones", "") or ""
        ).lower():
            pts.append(
                f"📋 **{ident}**: confirmar fase actual del "
                "plan de adaptación y próxima transición."
            )
    pts.extend([
        "💧 Estado del agua y bebederos (¿hubo hielo? ¿caudal OK?)",
        "🏠 Reparos / cama / barro en zona de comedero",
        "📦 Stock de Fibrogreen y demás insumos en campo",
        "📅 Próxima entrega coordinada / pendientes operativos",
    ])
    out["puntos_chequeo"] = pts
    return out


def generar_sugerencias_recordatorios() -> int:
    """Crea recordatorios automáticos basados en eventos del sistema:

    - **auto_lote_nuevo**: lote ingresado hace 5-7 días sin
      recordatorio del mismo tipo → llamar para chequear arranque.
    - **auto_cambio_fase**: lote con cambio de fase de adaptación
      en los próximos 1-3 días → llamar para coordinar transición.
    - **auto_sin_contacto**: cliente activo sin recordatorios
      completados en los últimos 30 días → llamar de control.

    Devuelve la cantidad de recordatorios creados.
    """
    from datetime import datetime as _dt, timedelta as _td
    hoy = _dt.now().date()
    creados = 0
    with get_conn() as conn:
        # ── 1. Lotes nuevos (5-7 días desde fecha_ingreso) ──
        try:
            lotes_nuevos = conn.execute(
                """SELECT l.cliente_id, l.identificador,
                          l.fecha_ingreso, c.nombre AS cli_nombre
                   FROM lotes l
                   JOIN clientes c ON c.id = l.cliente_id
                   WHERE l.estado = 'activo'
                     AND date(l.fecha_ingreso) BETWEEN
                         date(?, '-7 days') AND date(?, '-5 days')""",
                (hoy.isoformat(), hoy.isoformat()),
            ).fetchall()
        except Exception:
            lotes_nuevos = []
        for l in lotes_nuevos:
            if _existe_recordatorio_auto(
                l["cliente_id"], "auto_lote_nuevo", 14,
            ):
                continue
            try:
                conn.execute(
                    """INSERT INTO recordatorios_llamada
                       (cliente_id, fecha_objetivo, motivo, origen,
                        estado)
                       VALUES (?, ?, ?, 'auto_lote_nuevo',
                               'pendiente')""",
                    (
                        l["cliente_id"],
                        hoy.isoformat(),
                        (
                            f"Chequear arranque del lote "
                            f"'{l['identificador']}' "
                            f"(ingresado el {l['fecha_ingreso']}). "
                            f"Consumo, sanidad, comportamiento, "
                            f"adaptación al concentrado."
                        ),
                    ),
                )
                creados += 1
            except Exception:
                pass

        # ── 2. Cambio de fase próximo (próximos 3 días) ──
        # Requiere que el lote tenga plan de adaptación cargado.
        try:
            cambios_fase = conn.execute(
                """SELECT DISTINCT l.cliente_id, l.identificador,
                          d.id AS dieta_id
                   FROM lotes l
                   JOIN dietas d ON d.lote_id = l.id
                   WHERE l.estado = 'activo'
                     AND d.observaciones LIKE '%plan_adaptacion%'""",
            ).fetchall()
        except Exception:
            cambios_fase = []
        # Detección real de cambio próximo la hace el cron — acá
        # solo identificamos lotes con plan activo y dejamos que
        # el cron diario marque el día puntual. No generamos
        # sugerencia masiva acá para evitar ruido.

        # ── 3. Sin contacto hace > 30 días ──
        try:
            sin_contacto = conn.execute(
                """SELECT c.id AS cliente_id, c.nombre,
                          MAX(date(r.completado_en)) AS ultimo_llamado,
                          MAX(date(a.fecha)) AS ultima_alerta
                   FROM clientes c
                   LEFT JOIN recordatorios_llamada r
                          ON r.cliente_id = c.id
                         AND r.estado = 'hecho'
                   LEFT JOIN alertas_enviadas a
                          ON a.cliente_id = c.id
                   WHERE COALESCE(c.estado, 'activo') = 'activo'
                   GROUP BY c.id
                   HAVING (ultimo_llamado IS NULL
                           OR date(ultimo_llamado) <
                              date(?, '-30 days'))""",
                (hoy.isoformat(),),
            ).fetchall()
        except Exception:
            sin_contacto = []
        for s in sin_contacto:
            if _existe_recordatorio_auto(
                s["cliente_id"], "auto_sin_contacto", 21,
            ):
                continue
            try:
                conn.execute(
                    """INSERT INTO recordatorios_llamada
                       (cliente_id, fecha_objetivo, motivo, origen,
                        estado)
                       VALUES (?, ?, ?, 'auto_sin_contacto',
                               'pendiente')""",
                    (
                        s["cliente_id"],
                        hoy.isoformat(),
                        (
                            "Hace más de 30 días que no tenés "
                            "registro de llamado a este cliente. "
                            "Llamada de mantenimiento de relación: "
                            "consultar cómo va el lote, si tiene "
                            "novedades, próximas necesidades."
                        ),
                    ),
                )
                creados += 1
            except Exception:
                pass

    return creados


# =====================================================================
# DASHBOARD CACHE (blob JSON precomputado por cron externo)
# =====================================================================
#
# Contexto: el bloque de logística del dashboard corre un loop pesado
# (cliente × lote × día hasta agotamiento) que con Supabase remoto
# (200ms/query) tarda 30-60s. Para evitarle esa espera al asesor cada
# vez que abre la app en el campo, un cron externo (GitHub Actions
# cada 5 min) precalcula el resultado y lo guarda acá como JSON.
# La app solo hace 1 query de lectura (~200ms) y renderiza.
#
# Uso:
#   db.guardar_dashboard_cache('logistica_v1', {"filas_log": [...], ...})
#   data = db.leer_dashboard_cache('logistica_v1', max_edad_seg=900)
#   if data:  usar; else: recalcular on-demand.

# Bandera para no re-emitir el CREATE TABLE en cada llamada (evita
# roundtrip extra a Postgres). Se resetea al reiniciar el proceso.
_DASHBOARD_CACHE_INIT: dict = {"done": False}


def _ensure_dashboard_cache_table() -> None:
    """Crea la tabla `dashboard_cache` si no existe. Idempotente y
    válido para SQLite y Postgres.

    En Postgres la data va en JSONB (indexable, futuro-proof); en
    SQLite va en TEXT. En ambos casos guardamos/leemos como JSON
    string por simplicidad — el que quiera consultar por dentro del
    JSON puede usar `data::jsonb->>'campo'` en Postgres.
    """
    if _DASHBOARD_CACHE_INIT["done"]:
        return
    try:
        if _usando_postgres():
            with get_conn() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS dashboard_cache (
                           id TEXT PRIMARY KEY,
                           data JSONB NOT NULL,
                           updated_at TIMESTAMPTZ NOT NULL
                               DEFAULT NOW()
                       )"""
                )
        else:
            with get_conn() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS dashboard_cache (
                           id TEXT PRIMARY KEY,
                           data TEXT NOT NULL,
                           updated_at TEXT NOT NULL
                               DEFAULT CURRENT_TIMESTAMP
                       )"""
                )
        _DASHBOARD_CACHE_INIT["done"] = True
    except Exception:
        # No romper la app si por alguna razón no se puede crear
        # (permisos, connection error, etc). El cliente detectará
        # que leer_dashboard_cache devuelve None y caerá al camino
        # on-demand.
        pass


def guardar_dashboard_cache(id: str, data: Dict) -> None:
    """Guarda (upsert) el blob JSON `data` bajo la clave `id`.

    Marca `updated_at = now()` (server-side) — así la frescura no
    depende del reloj del cliente que llama.
    """
    _ensure_dashboard_cache_table()
    payload = json.dumps(data, ensure_ascii=False, default=str)
    with get_conn() as conn:
        if _usando_postgres():
            # ON CONFLICT necesita id como PK — ya lo es.
            # Cast explícito a jsonb — sin él Postgres implica-casta
            # pero el error es feo cuando el JSON tiene comillas
            # raras. Con el cast, cualquier JSON válido pasa.
            conn.execute(
                """INSERT INTO dashboard_cache (id, data, updated_at)
                   VALUES (?, ?::jsonb, NOW())
                   ON CONFLICT (id) DO UPDATE
                     SET data = EXCLUDED.data,
                         updated_at = NOW()""",
                (id, payload),
            )
        else:
            # SQLite: UPSERT con ON CONFLICT (v3.24+). Usar
            # datetime('now') server-side (UTC) para que coincida con
            # julianday('now') al leer — datetime.now() en Python es
            # local y desfasa 3h en AR.
            conn.execute(
                """INSERT INTO dashboard_cache (id, data, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(id) DO UPDATE
                     SET data = excluded.data,
                         updated_at = datetime('now')""",
                (id, payload),
            )


def leer_dashboard_cache(
    id: str, max_edad_seg: int = 900,
) -> Optional[Dict]:
    """Devuelve el blob JSON guardado bajo `id` si tiene menos de
    `max_edad_seg` segundos. None si no existe o está viejo.

    El dict devuelto tiene una clave extra `_edad_seg` con la edad
    del snapshot en segundos, para que la UI pueda mostrar
    "datos de hace X min".
    """
    _ensure_dashboard_cache_table()
    try:
        with get_conn() as conn:
            if _usando_postgres():
                r = conn.execute(
                    """SELECT data::text AS data,
                              EXTRACT(EPOCH FROM (NOW() - updated_at))
                                  AS edad_seg
                       FROM dashboard_cache
                       WHERE id = ?""",
                    (id,),
                ).fetchone()
            else:
                r = conn.execute(
                    """SELECT data,
                              (julianday('now')
                                  - julianday(updated_at)) * 86400
                                  AS edad_seg
                       FROM dashboard_cache
                       WHERE id = ?""",
                    (id,),
                ).fetchone()
    except Exception:
        return None
    if not r:
        return None
    try:
        edad = float(r["edad_seg"] or 0)
    except Exception:
        edad = 0.0
    if edad > max_edad_seg:
        return None
    try:
        data = json.loads(r["data"])
    except Exception:
        return None
    if isinstance(data, dict):
        data["_edad_seg"] = edad
    return data
