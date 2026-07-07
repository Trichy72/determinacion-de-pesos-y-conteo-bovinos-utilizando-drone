# Migración del sistema HMS a la nube — Plan de trabajo

Este documento describe cómo mover el sistema HMS Nutrición Animal
desde la Mac local a la nube, **sin cortar el servicio a los clientes
en ningún momento**. La operación (emails, WhatsApp, informes) sigue
funcionando el 100 % del tiempo durante toda la migración.

## Objetivo

Poder operar desde cualquier computadora (Mac, PC, tablet en el
campo), apagar la Mac cuando querramos, y mantener el costo mensual lo
más bajo posible.

## Arquitectura final elegida

| Componente actual (Mac) | Componente nuevo (nube)     | Costo mensual   |
| ----------------------- | --------------------------- | --------------- |
| Streamlit local         | Streamlit Community Cloud   | USD 0           |
| SQLite (`data/*.db`)    | Supabase (Postgres)         | USD 0 (500 MB)  |
| launchd crons           | GitHub Actions              | USD 0 (2000 min)|
| Fotos en `data/`        | Cloudflare R2               | USD 0 (10 GB)   |
| Credenciales locales    | Secrets de cada plataforma  | USD 0           |

**Costo total esperado el primer año: USD 0.**
Cuando pasemos los cupos gratuitos (~1 año), estimado USD 15-25/mes.

## Regla de oro

En ningún momento va a haber dos sistemas mandando el mismo mensaje al
mismo cliente. Cada cron se migra con la Mac como "backup" hasta que
el nuevo demuestre que funciona igual.

## Paso 1 — Red de seguridad (HECHO)

- [x] Backup fechado de la base y credenciales en `backups/`.
- [x] Script `backups/restaurar_backup.sh` para rollback en 1 minuto.
- [x] `.gitignore` que protege credenciales y datos privados.
- [x] `.env.example` con la plantilla de todas las variables.
- [x] Este documento (`MIGRACION_NUBE.md`).

## Paso 2 — Subir código a GitHub (pendiente, tarea del usuario)

1. Crear cuenta en <https://github.com> si no tenés.
2. Crear un repo **privado** llamado `hms-nutricion` (no público —
   aunque el `.gitignore` protege las credenciales, mejor privado).
3. Desde la terminal, en la carpeta del proyecto, correr:

   ```bash
   cd "/Users/hms/Documents/Claude/Projects/determinacion de pesos y conteo bovinos utilizando drone"
   git init
   git add .
   git commit -m "Estado inicial antes de migración a nube"
   git branch -M main
   git remote add origin https://github.com/<TU-USUARIO>/hms-nutricion.git
   git push -u origin main
   ```

4. Verificar en <https://github.com/TU-USUARIO/hms-nutricion> que
   estén los archivos pero NO estén `data/cattle_tracker.db` ni los
   `*_config.json`.

## Paso 3 — Migrar la base a Supabase (Postgres)

- Crear cuenta en <https://supabase.com> (gratis).
- Crear proyecto `hms-nutricion` (elegir región São Paulo por latencia).
- Anotar la `DATABASE_URL` que da Supabase.
- Ejecutar el script de migración (a crear) `scripts/migrar_sqlite_a_postgres.py`.
- Adaptar `src/database.py` para hablar Postgres (usar SQLAlchemy o
  pg8000 con schema idéntico).
- **La Mac apunta a Supabase.** Cero cambios visibles, pero ahora los
  datos viven en la nube. Este es el hito clave: a partir de acá, todo
  lo demás se puede migrar sin miedo.

## Paso 4 — App web en Streamlit Cloud (en paralelo)

- Conectar Streamlit Cloud al repo de GitHub.
- Cargar los secretos (SMTP, WhatsApp, Anthropic, WeatherAPI,
  DATABASE_URL de Supabase).
- Deploy. Va a aparecer en `https://hms-nutricion.streamlit.app`.
- La app local y la de la nube leen la MISMA base. Podés usar
  cualquiera.

## Paso 5 — Fotos a Cloudflare R2

- Crear cuenta Cloudflare (gratis).
- Crear bucket `hms-fotos-lote`.
- Refactor de `src/fotos_lote.py` y `src/pdf_lote_inspeccion.py` para
  usar boto3/S3 en vez de rutas locales.
- Subir fotos históricas por batch.
- Nueva subida va directo a R2.

## Paso 6 — Crons a GitHub Actions

- Un `.yml` por cron. Ejemplos:
  - `.github/workflows/alertas_manana.yml` (cron `0 11 * * *` UTC = 8 AR)
  - `.github/workflows/alertas_tarde.yml`
  - `.github/workflows/informe_semanal.yml`
  - `.github/workflows/silocomedero.yml`
  - `.github/workflows/demanda_semanal.yml`
- Cada workflow:
  1. Clona el repo
  2. Instala requirements
  3. Corre el script correspondiente con env vars de secrets
- **Fase de prueba silenciosa** (1 semana por cron):
  - GitHub Actions genera el mensaje pero con flag `DRY_RUN=true`.
  - Comparamos con lo que efectivamente mandó la Mac.
- **Cuando coinciden**: desactivar el launchd correspondiente, sacar
  el `DRY_RUN` y GitHub Actions envía real.
- Cron por cron. Nunca los 8 juntos.

## Paso 7 — Apagar la Mac

Sólo cuando pase al menos 1 semana con **todos** los crons en GitHub
Actions funcionando bien sin la Mac.

## Rollback

En cualquier momento, si algo falla:

1. Reactivar el launchd de la Mac correspondiente:
   `launchctl load ~/Library/LaunchAgents/com.hms.<cron>.plist`
2. Desactivar el workflow de GitHub Actions correspondiente
   (interfaz web, `.github/workflows/*.yml.disabled`).
3. Si es más grave, correr `./backups/restaurar_backup.sh`.

## Timing estimado

| Paso | Trabajo mío | Tu participación | Duración calendario |
|------|-------------|------------------|---------------------|
| 1    | 30 min      | 0                | HECHO               |
| 2    | 15 min      | 30 min (crear cuenta + push) | 1 día  |
| 3    | 2 días      | 15 min (crear cuenta Supabase) | 3-4 días |
| 4    | 1 día       | 30 min (crear cuenta Streamlit) | 1 día  |
| 5    | 1 día       | 15 min (crear cuenta Cloudflare)| 1 día |
| 6    | 1 día setup + 1 semana observación | verificar que llegan emails | 1-2 semanas |
| 7    | -           | apagar Mac                       | -        |

**Total: 3-4 semanas de calendario con operación 100 % activa.**
