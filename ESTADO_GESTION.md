# Estado del proyecto — App de gestión (HMS Gestión Ganadera)

> **Para Claude:** este archivo es tu memoria de este frente de trabajo.
> Leelo completo antes de tocar nada. Al terminar la sesión, actualizalo:
> qué se hizo, qué decisiones se tomaron y por qué, y qué quedó pendiente.
> No borres el historial de decisiones: sirve para no volver a discutir lo
> mismo. Si una decisión cambia, dejá la vieja y anotá por qué cambió.

**Última actualización:** 29/07/2026

---

## Qué es

App web de gestión para HMS Nutrición Animal (Mauricio Suárez, asesor
nutricional a campo, La Pampa / Pampa Húmeda). La usa Mauricio, no los
clientes: es su centro de comando.

- **Online:** https://hms-gestion-ganadera.streamlit.app
- **Repo GitHub:** Trichy72, carpeta local
  `~/Documents/Claude/Projects/determinacion de pesos y conteo bovinos utilizando drone`
- **Base de datos:** Postgres en Supabase (región Oregon)
- **Archivo principal:** `app.py` (~15.900 líneas, monolito Streamlit)

## Pestañas

| Pestaña | Qué hace |
|---|---|
| Inicio | Dashboard: prioridades del día, tarjetas de clientes con semáforo, agenda, clima |
| Clientes/Lotes | CRUD de clientes y lotes, ficha clínica, cargas de silo y rollo, entregas |
| Evolución | ADG, conversión alimenticia y uniformidad a partir de las pesadas guardadas |
| Análisis | Uniformidad de una pesada con pesos individuales + optimizador de dietas (LP) |
| Asesor IA | Chat con Claude, con acceso al histórico completo del lote |
| Historial | Timeline por lote |
| Configuración | Identidad HMS, API key, SMTP, WhatsApp, ingredientes |
| Ayuda | Cómo está armado el sistema |

## Decisiones tomadas (no volver a discutir sin motivo)

- **Un solo repo para las dos apps.** El drone y la gestión comparten
  `src/database.py`, que es el nexo (el drone guarda pesadas en la ficha
  del cliente). Separar en dos repos obligaría a duplicar la base y las
  copias se desincronizan. Decidido el 29/07/2026.
- **`requirements.txt` es la versión "lite"** (sin ultralytics/torch/cv2)
  para que la app entre en el free tier de Streamlit. El completo es
  `requirements-full.txt`, para la Mac.
- **`app.py` no contiene nada de procesamiento de imágenes.** Todo eso
  vive en `drone_app.py` (local). Si algo necesita YOLO u OpenCV, va en la
  app local, no acá.
- **Dashboard precomputado:** un cron calcula el blob y la app lo lee, con
  tolerancia de 6 horas. Sin esto el dashboard tardaba 30-60 s.

## Pendientes

| # | Tarea | Notas |
|---|---|---|
| — | Commitear y pushear el fix de `SMTP_BCC_CLIENTES` | El bug sigue vivo en producción hasta el push |
| — | `CARGA_BASE_URL` apunta a un túnel ngrok free de la Mac | Bloqueante real del 482: la URL muere al reiniciar ngrok |
| 485 | Migrar Supabase a São Paulo | Bajaría la latencia de ~200 ms a ~50 ms |
| 479 | Fotos de inspección a Cloudflare R2 | Hoy van a la base, no escala |
| 481 | Fase de sombra: crons en la nube + Mac en paralelo con DRY_RUN | Antes de apagar la Mac |
| 482 | Apagar la Mac (fin de la migración) | Depende de 481 |
| 284 | Integrar dieta completa + sistema de comedero a las alertas | En curso |
| — | Rotar credenciales expuestas (SMTP Brevo, Twilio, WeatherAPI) | Backlog de seguridad |
| — | Crear Roxdan y La Esperanza Argentina como clientes | Para poder guardar las recorridas del drone en ficha |

## Cosas que conviene saber antes de tocar

- **El índice de git se ensucia solo.** El trabajo con índice temporal deja
  un `.git/index.lock` de tamaño 0 que el mount FUSE no puede borrar, y el
  `.git/index` queda desincronizado (el 29/07 tenía 12 borrados staged,
  entre ellos los dos `ESTADO_*.md`; en HEAD estaban todos). Se arregla
  moviendo el lock a `_to_delete/` y corriendo `git read-tree HEAD`, que no
  toca el working tree. Conviene mirar `git status` antes de pushear.
- **Streamlit Cloud corre Python 3.14** (App settings → General). Tenerlo en
  cuenta si una dependencia no tiene wheel para esa versión.

- **Git en esta carpeta:** es un mount FUSE donde `unlink` está prohibido.
  Los comandos normales de git fallan. Se commitea con plumbing:
  `GIT_INDEX_FILE=/tmp/x git read-tree $(cat .git/refs/heads/master)` →
  `git update-index --add <archivos>` → `git write-tree` →
  `git commit-tree` → escribir el hash en `.git/refs/heads/master`.
  Los warnings de "unable to unlink" son normales, se ignoran.
- **El push va por GitHub Desktop** (botón "Push origin"), no por línea de
  comandos.
- **Streamlit ≥1.49:** las pestañas usan react-aria, no BaseWeb. Los
  selectores CSS son `div[data-testid="stTabs"]`, `[role="tablist"]`,
  `div[data-testid="stTab"]`. `.stTabs` y `data-baseweb` ya no existen.
- **Verificar antes de dar por bueno:** `python3 -m py_compile app.py` no
  alcanza. Conviene correr el análisis AST de nombres huérfanos (busca
  variables usadas que ya no existen) y probar la lógica nueva contra una
  base SQLite temporal.

## Historial de sesiones

### 29/07/2026 (tarde) — Cerrar el 484 y un bug de BCC en la nube

- **Los 22 secrets ya estaban cargados en Streamlit Cloud.** El pendiente
  484 estaba viejo. Verificado comparando hash SHA-256 de cada valor contra
  `.env` + `data/*.json` de la Mac: los 22 coinciden. La app en la nube
  levanta bien (5 clientes, 173 animales, clima y Asesor IA operativos).
- **Bug encontrado y arreglado en `src/alertas_email.py`.**
  `_cargar_smtp_desde_env` leía `SMTP_BCC_CLIENTES` con el helper `_b()`,
  o sea como booleano, pero `enviar_email` trata `bcc_clientes` como
  dirección de email y la mete en `all_recipients`. Medido:
  con `"true"` los destinatarios quedaban `['cliente@…', 'True']` — el
  servidor rechaza ese RCPT pero acepta el del cliente, así que `sendmail`
  no levanta excepción y la función devuelve `True, "Email enviado"`;
  con un email real devolvía `False` y no mandaba BCC. En los dos casos
  Mauricio perdía su copia oculta sin ningún error visible. Ahora un helper
  `_bcc()` acepta email, lista con comas, o booleano cayendo a
  `SMTP_ADMIN_EMAIL`. **Todavía sin pushear: en producción el bug sigue.**
- **`.streamlit/secrets.toml.example` reescrito.** Tenía nombres que el
  código no lee (`SMTP_FROM`, `SMTP_BCC_ADMIN`, `IMAP_PORT`,
  `TWILIO_WHATSAPP_FROM`, `TWILIO_ADMIN_WHATSAPP`) y le faltaban
  `CARGA_BASE_URL`, `TWILIO_MODO_SANDBOX` y los `SMTP_USE_*`. Cargar los
  secrets siguiendo ese archivo dejaba WhatsApp muerto. Sus 22 claves ahora
  son idénticas a las de `.github/workflows/`. `.env.example` sigue con los
  nombres viejos, pero ese es para la Mac y ahí manda el JSON.
- La doc decía puerto 465 con SSL; la config real es Brevo en 587 con
  STARTTLS. Los valores de verdad salen de `data/smtp_config.json`.
- Índice de git limpiado (ver "Cosas que conviene saber").
- Generado `data/.secrets_streamlit.toml` (permiso 600, ignorado por
  `data/.*`) como copia local del TOML de la nube.

### 29/07/2026 — Separar el drone de la gestión
- Renombrado el repo y la URL (antes decía "determinación de pesos y
  conteo bovinos utilizando drone", que no describía la app).
- Sacado de `app.py`: import de cv2, stubs del drone, `load_detector`,
  `CattleDetector`, `WeightModel`, `process_video`. −179 líneas.
- **Evolución** pedía subir dos videos: imposible en la nube. Reescrita
  para leer las pesadas guardadas del lote (drone o balanza) y calcular
  ADG, conversión alimenticia y cambio de CV.
- **Análisis** leía `st.session_state["vid_animales"]`, que nadie escribía
  desde que el video se procesa en la Mac: siempre mostraba "no hay
  resultados". Reescrita para analizar los pesos individuales de una
  pesada guardada.
- **Bug corregido en `guardar_pesada`:** reventaba con TypeError si
  `desvio_kg` venía en None (pesada de balanza donde solo se anota el
  promedio). Ahora `cv_pct` queda NULL = "desconocido", distinto de 0 =
  "lote perfectamente parejo".
- Actualizada la pestaña Ayuda (hablaba de marcadores ArUco de 1,02 m, que
  ya no se usan: la altura sale del EXIF de la foto).
