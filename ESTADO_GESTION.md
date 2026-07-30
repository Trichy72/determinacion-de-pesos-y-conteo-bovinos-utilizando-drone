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
| — | El dashboard sigue mostrando "Reponer HOY" con historial incompleto | `dashboard_precompute.py:129` solo excluye `sin_entregas` |
| — | Umbrales de stock configurables desde Configuración | Hoy están escritos a mano en 4 lugares con valores distintos |
| — | `backups/restaurar_backup.sh` quedó obsoleto | Restaura a `data/cattle_tracker.db`: es de la era SQLite, no sirve para Postgres |
| — | Integración con facturación: crear rol `hms_ganadera_ro` | SQL listo; falta correrlo y cargar `FACTURACION_DATABASE_URL` |
| 485 | Migrar Supabase a São Paulo | Bajaría la latencia de ~200 ms a ~50 ms |
| 479 | Fotos de inspección a Cloudflare R2 | Hoy van a la base, no escala |
| 481 | Fase de sombra: crons en la nube + Mac en paralelo con DRY_RUN | Antes de apagar la Mac |
| 482 | Apagar la Mac (fin de la migración) | Depende de 481 |
| 284 | Integrar dieta completa + sistema de comedero a las alertas | En curso |
| — | Rotar credenciales expuestas (SMTP Brevo, Twilio, WeatherAPI) | Backlog de seguridad |
| — | Crear Roxdan y La Esperanza Argentina como clientes | Para poder guardar las recorridas del drone en ficha |

## Cosas que conviene saber antes de tocar

- **Fecha de corte de stock.** `ajustes_stock` (cliente, lote, producto,
  fecha, kg) le pone un piso al cálculo: con un ajuste vigente, el stock
  arranca de los kg declarados, suma solo las entregas POSTERIORES y
  acumula consumo desde esa fecha. Las entregas con la MISMA fecha del
  corte se consideran ya incluidas en lo declarado, y eso se reporta en
  `kg_entregas_en_fecha_corte` para avisarlo en pantalla — la ambigüedad
  (¿contó antes o después de que llegara el camión?) no la resuelve
  ninguna regla, así que se muestra en vez de decidirla en silencio.
  Con corte, un 0 kg SÍ es agotamiento real y dispara alerta.

- **El repo es PÚBLICO.** Nada de dumps, artifacts con datos, ni
  credenciales. Revisado el 30/07/2026: el historial de git está limpio
  (busqué la clave de Brevo, el token de Twilio, la de WeatherAPI, la
  contraseña de la base y la de Anthropic en todos los commits — cero
  coincidencias; el único `sk-ant-api03-...` es un placeholder de
  plantilla). El pendiente de "credenciales expuestas" no viene de git.
- **Supabase plan Free no hace backups.** El panel dice "No backups".
  De eso se ocupa `.github/workflows/backup_db.yml` desde el 30/07/2026.
- **El servidor es Postgres 17.6**, así que `pg_dump` tiene que ser 17 o
  más nuevo. `ubuntu-latest` trae el 16 y falla, por eso el workflow
  instala `postgresql-client-17` del repo oficial.
- **La base pesa 12 MB** (18 tablas; las más grandes son
  `alertas_enviadas` con 200 kB y `alertas_whatsapp_enviadas` con 136 kB).
  Por eso el dump entra cómodo en un adjunto de email.

- **El stock NO es un saldo de depósito, es un balance acumulado.**
  `kg_restantes = max(0, todas_las_entregas − consumo_teórico_desde_la_primera_entrega)`.
  Si el lote venía comiendo desde antes de la primera entrega cargada en
  el sistema, arrastra un déficit permanente y la barra queda clavada en
  0 hasta que lo cubrís. No hay stock inicial ni fecha de corte: cargar
  entregas nuevas no "resetea" nada.
- **Los umbrales de stock están escritos a mano en 4 lugares y no
  coinciden:** semáforo `≤0` y `≤7` (`app.py:4131`), tarjeta de
  prioridades `≤0` (`app.py:3599`), alerta a clientes `14`
  (`alertas_diarias.py:1179`), fin de carga de silo `1`
  (`alertas_diarias.py:1318`).
- **Para el análisis AST de nombres huérfanos ahora hay script:**
  `python3 scripts/nombres_huerfanos.py app.py src/*.py`. Hace scoping
  léxico de verdad (funciones anidadas, lambdas, comprehensions), así
  que no tira los falsos positivos de un walk ingenuo. Con `--json`
  compara antes/después: lo útil es que el diff no traiga huérfanos
  nuevos, no que el total sea cero.

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

### 30/07/2026 (tarde) — Fecha de corte de stock

Cierra el problema que quedó abierto anoche: la barra clavada en cero y
los lotes mudos.

  - `ajustes_stock` con DDL portable (`SERIAL` en Postgres,
    `AUTOINCREMENT` en SQLite) creada de forma perezosa, porque
    `init_db` sale temprano cuando la base es Postgres y en la nube
    nadie la crearía. Más `crear_ajuste_stock`, `listar_ajustes_stock`
    y `borrar_ajuste_stock`.
  - `calcular_stock_actual` toma el ajuste vigente (el más reciente con
    fecha <= la de referencia) y arranca de ahí. Con corte,
    `deficit_historial` es 0 por definición: no hay historial que
    inferir.
  - Pantalla en la ficha del cliente, abajo de la tabla de stock, con
    columna "Corte" y listado de cortes cargados.
  - **Bug propio encontrado y corregido en el camino:** la primera
    versión descartaba las entregas con la misma fecha del corte, así
    que cargar bolsas el día del recuento no movía la barra — el mismo
    síntoma que veníamos a arreglar. Ahora se reportan aparte y se
    avisan en pantalla.
  - Verificado contra SQLite (sin ajuste no cambia nada; con ajuste la
    barra se mueve; las entregas previas al corte no se cuentan dos
    veces; el ajuste más nuevo manda; un agotamiento real después del
    corte sí alerta) y la rama Postgres del DDL contra un Postgres 16
    local (create idempotente, SERIAL, `lote_id` nulo, y la consulta de
    listado).

### 30/07/2026 — Backup de la base, y el camino a la facturación

**Backup (hecho).** La base de producción no tenía ningún respaldo: el
único backup del repo es del 07/07 y es de la era SQLite
(`cattle_tracker.db` + credenciales), con un `restaurar_backup.sh` que
apunta a `data/cattle_tracker.db`. Desde la migración a Supabase habían
pasado 23 días sin respaldo en ningún lado.

  - `scripts/backup_db.py`: `pg_dump -Fc`, **verificación**, y envío por
    email al admin reusando la config SMTP existente (cero credenciales
    nuevas). La verificación es el punto: lista el dump con
    `pg_restore -l` y exige que aparezcan `clientes`, `lotes`, `dietas`,
    `pesadas` y `entregas_producto`. Un dump truncado también existe y
    también pesa bytes; sin esto el cron podría escribir basura durante
    meses sin que nadie se enterara.
  - `.github/workflows/backup_db.yml`: diario 03:00 AR, con un paso que
    manda mail si el backup falla. **No sube artifacts ni commitea el
    dump: el repo es público.**
  - Implementado `attachments=` en `enviar_email`. El docstring del
    módulo lo documentaba desde siempre pero el parámetro no existía:
    llamarlo tiraba `TypeError`.
  - Probado de punta a punta contra un Postgres local: dump válido OK,
    dump truncado rechazado, archivo vacío rechazado, dump sin la tabla
    `pesadas` rechazado nombrándola, y el adjunto verificado dentro del
    MIME.

**Facturación (relevado, sin implementar).** El ERP comercial es una SPA
en Netlify (`spectacular-paletas-150f49.netlify.app`) con backend en un
**segundo proyecto Supabase**, `srluebpighcafbgxbuwy`. No es FastAPI:
esa nota estaba mal.

  - Esquema: `comprobantes`, `clientes`, `productos`, `movimientos`,
    `pedidos`, `zonas`, `app_config` — todas con la misma forma
    (`id`, `user_id`, `payload jsonb`, `updated_at`). Todo el contenido
    vive en `payload`, que siendo jsonb se consulta con SQL normal.
  - `facturas_pub` (`id`, `data jsonb`) es la única tabla legible con la
    key pública, y tiene **solo las facturas publicadas a mano** (7 al
    30/07). No es el archivo.
  - En `comprobantes` hay **24 facturas, desde el 13/04/2026**. Lo
    anterior quedó en Dux. Los renglones están en
    `payload->'items'` como `{cant, desc, unidad, precio}`, con `unidad`
    tipo `"BOLSA 30 KG"` — o sea kg calculables. `payload->>'clienteId'`
    apunta a `clientes.id`, y ahí está el `cuit`.
  - El puente con esta app es el **CUIT**, que hay que agregar a
    `clientes` (hoy no existe). Confirmado que hace falta: "Ezequiel
    Pezzola" y "Pedro Manuel Pezzola" son dos CUIT distintos
    (20271032553 y 23121852969) pero por nombre se confunden.
  - Ojo al importar: en una misma factura conviven `FIBROTER X 30 KG` y
    `CONCENTRADO PARRILLEROS X 25 KG`. Solo deben generar entrega los
    productos que están en la dieta del lote. Y hay clientes sin CUIT
    (LECCEA, MALATESTA, PEIRETTI) y uno con CUIT inválido de 6 dígitos
    (MOYA CLAUDIO, `554445`): el matcher tiene que validar 11 dígitos y
    avisar, no tragárselos.

### 29/07/2026 (noche) — Falsos positivos de stock a clientes

Mauricio avisó que a clientes reales les está llegando el mail de stock
bajo, y que cargó bolsas y la barra de stock no se movió. **Es el mismo
bug las dos cosas.**

- **Causa.** `calcular_stock_actual` hace
  `max(0, entregas − consumo_teórico_desde_la_primera_entrega)`. Cuando
  el lote venía comiendo desde antes de la primera entrega cargada, el
  balance da negativo y el `max()` lo tapa con un 0 que es
  indistinguible de "se quedó sin producto". Reproducido en SQLite: lote
  con primera entrega hace 60 días y 1.000 kg, consumo 108 kg/día →
  déficit 5.480 kg; cargar 10 bolsas (300 kg) y después 40 más
  (1.200 kg) deja la barra en 0 kg las tres veces. Hay que cargar más de
  6.480 kg para que se mueva un kilo.
- Ese 0 es el que `clientes_con_stock_bajo(14)` lee como 0 días y manda
  al cliente. El código ya excluía `sin_entregas`, pero el historial
  incompleto devolvía `sub_uso` y se colaba.
- **Fix.** Nuevo diagnóstico `historial_incompleto` (con
  `deficit_historial_kg` en el dict de retorno) cuando el consumo
  acumulado supera lo entregado, y `clientes_con_stock_bajo` lo excluye
  igual que `sin_entregas`. Verificado con 3 casos: historial incompleto
  excluido, stock bajo real sigue avisando, stock holgado no avisa.
  Icono 🟣 en la tabla de stock para que se vea el estado.
- **Lo que el fix NO hace:** no arregla el cálculo, solo deja de avisarle
  al cliente por un dato que no es confiable. La barra sigue en 0 hasta
  que exista stock inicial / fecha de corte (pendiente nuevo).
- Agregado `scripts/nombres_huerfanos.py`, que faltaba: este archivo
  mandaba a correr el análisis AST pero el script no existía.

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
