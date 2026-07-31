# Estado del proyecto — App de gestión (HMS Gestión Ganadera)

> **Para Claude:** este archivo es tu memoria de este frente de trabajo.
> Leelo completo antes de tocar nada. Al terminar la sesión, actualizalo:
> qué se hizo, qué decisiones se tomaron y por qué, y qué quedó pendiente.
> No borres el historial de decisiones: sirve para no volver a discutir lo
> mismo. Si una decisión cambia, dejá la vieja y anotá por qué cambió.

**Última actualización:** 31/07/2026

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
| — | **Corregir el `kg_tal_cual` de Fibroter en Ezequiel Pezzola** | Dice 0,400 y a campo son 1,10. Es dato, no código: el `pct_ms` de 33% sí es correcto |
| — | Revisar las dietas con MS implícita > 100% | Jackie 177%, Pedro 112%, Mario 111%. El DMI incluye forraje que la composición no lista o lista con `pct_ms = 0` |
| — | Cargar la fila de forraje en el lote de Jackie Graves | Lote "forraje aparte" sin ninguna fila de forraje: el DMI de 5,32 incluye ~2,7 kg MS de pastoreo que no está en la composición |
| — | Enchufar `factor_escala_consumo_pv` en el camino del stock | Existe y lo usa la carga de silo, pero `calcular_consumo_diario_kg` no. Explica el 3-8% que falta |
| — | Descontar del ADPV objetivo la pérdida por frío ya calculada | `impacto_productivo` la calcula y `impactos_lote` la guarda, pero la proyección de peso sigue usando el objetivo plano |
| — | Sumar el plus de mantenimiento por frío al consumo | `dmi.py` lo calcula. Es el efecto opuesto al anterior: hay que aplicar los dos o ninguno |
| — | **Ver andando el gráfico de evolución del peso** | Nunca se renderizó: se escribió en un entorno sin altair. Mirarlo antes de darlo por bueno |
| — | Commitear y pushear el gráfico (`serie_peso.py` + `app.py`) | Sin pushear al cierre del 31/07 |
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

- **NUNCA escribir un test que cree datos sin pasar por
  `scripts/_sandbox_db.py`.** El 31/07/2026 dos tests escribieron en la
  base de PRODUCCIÓN y le dejaron 28 clientes de prueba, cinco de ellos
  duplicando el nombre de clientes reales. Hacían
  `os.environ.pop("DATABASE_URL")` dando por sentado que sin esa
  variable el backend cae a SQLite. **No es así:**
  `db_backend._get_database_url()` (línea 45), cuando no encuentra la
  variable de entorno, **lee el `.env` del proyecto directamente**. El
  pop no sirve de nada. `_sandbox_db.base_temporal()` interviene el
  resolutor y **después verifica** que la conexión no sea Postgres,
  abortando el proceso si lo es. Dos lecciones que costaron caro:
  sacar la variable de entorno NO alcanza, y **un test que escribe en
  producción no avisa: pasa en verde**. Los 30/30 y 22/22 de esa
  corrida eran correctos y destructivos al mismo tiempo.
- **La tabla `clientes` en Postgres NO tiene el `UNIQUE` sobre
  `nombre`** que sí declara el DDL de SQLite (porque `init_db` sale
  temprano en Postgres y el esquema de la nube se creó por otro
  camino). Por eso se pudieron crear cinco "Ezequiel Pezzola". Si
  alguna vez se cruza por nombre, tenerlo presente: el nombre no es
  clave. El puente bueno es el CUIT.
- **La limpieza quedó en `scripts/limpiar_datos_de_test.py`**, por si
  vuelve a hacer falta el patrón: lista primero, borra solo con
  `--confirmar`, y protege por `fecha_alta` — nunca por patrón de
  nombre, que se llevaría puesto un cliente real homónimo.
- **Streamlit Cloud NO deja cambiar el repo de una app existente.**
  Revisado en pantalla el 31/07: App settings solo tiene General
  (subdominio y versión de Python), Sharing y Secrets. Para re-apuntar
  la app al nombre nuevo habría que borrarla y recrearla, o sea volver
  a cargar los 22 secrets a mano. Conclusión: **el reboot manual
  después de cada push se queda**, es el costo de haber renombrado el
  repo. El subdominio ya es el correcto y corre Python 3.14.

- **DECISIÓN DE MODELO (31/07/2026): el consumo diario de producto sale
  de `kg_tal_cual` de la dieta, NO de `DMI × pct_ms / 100`.** La fórmula
  vieja devolvía kg de MATERIA SECA, y todo lo que hay del otro lado de
  la resta — entregas, bolsas de 30 kg, stock — está en producto TAL
  CUAL: se restaban unidades distintas. Además `consumo_ms_kg` y
  `pct_ms` no cierran con `kg_tal_cual` en la mayoría de las dietas
  reales, porque el DMI incluye forraje que la composición no lista.
  **La prueba: la materia seca implícita** (`DMI × pct_ms / 100 /
  kg_tal_cual`) **daba 217%, 177%, 112%, 111% y 88% en los 5 lotes de
  producción.** Arriba de 100% es físicamente imposible. Contra los
  valores que Mauricio validó a campo, `kg_tal_cual` se desvía 3-8% y
  la fórmula vieja se desviaba −21% a +72%. `calcular_consumo_diario_kg`
  ahora devuelve `ms_implicita_pct` justamente para poder detectar
  dietas mal cargadas sin tener que ir a mirarlas de a una.

- **El consumo TIENE que escalar con el peso vivo** (dicho por Mauricio,
  30 y 31/07): se proyecta la evolución del peso por días y ganancia, y
  el consumo se actualiza como un porcentaje del peso vivo según la
  categoría; después esa proyección se ajusta con la balanza o con la
  estimación del drone. La pieza ya existe —
  `factor_escala_consumo_pv`— pero **solo la usa el camino de la carga
  de silocomedero**. El camino del stock y las alertas todavía no, y
  ahí está el 3-8% que falta después de pasar a `kg_tal_cual`.

- **Y la ganancia de peso varía con el clima** (Mauricio, 31/07), que es
  lo que explica que la proyección se fuera 17-27% arriba en los CINCO
  lotes a la vez: cinco lotes errando para el mismo lado no es
  casualidad, es una variable común, y los lotes son de mayo — pleno
  invierno en La Pampa. **Son dos efectos opuestos y hay que aplicar
  los dos o ninguno:** con frío el animal gasta más en mantenimiento y
  COME MÁS por kilo de peso vivo, pero GANA MENOS y por lo tanto pesa
  menos que lo proyectado. Corregir solo por peso sobreestima;
  corregir solo por mantenimiento subestima. El neto es ese 3-8%.
  Las dos piezas ya están escritas y ninguna llega a la proyección de
  peso: `dmi.py` (DMI ajustado por frío, calor, humedad, viento, barro,
  pelaje mojado y acumulación; `dmi_base_kg` es literalmente "% del PV
  según categoría") e `impacto_productivo.py::estimar_impacto_frio`,
  que devuelve `adpv_perdida_kg_rango` — cuántos kg/día de ganancia
  perdió el lote en cada evento. La tabla `impactos_lote` guarda ese
  historial con `clima_resumen_json` y estado "confirmado (recalculado
  con clima histórico real)".

- **DECISIÓN DE MODELO (30/07/2026): el stock de producto se lleva por
  CLIENTE + PRODUCTO, no por lote.** Mauricio lo confirmó preguntándole
  cómo es en el campo: *"todo se junta en un mismo lado y se reparte a
  los distintos lotes según la ración preparada para ese lote"*. O sea
  que el stock por lote era una división que en la realidad no existe.
  Los lotes y las dietas SE QUEDAN: son los que dan el consumo diario;
  el consumo de un producto es la suma de los lotes que lo comen. La
  consecuencia importante: la importación desde facturación no tiene que
  adivinar a qué lote va cada renglón de factura, porque no hace falta
  saberlo. Si alguna vez se vuelve a poner stock por lote, hay que poder
  responder de dónde sale el reparto — y hoy no sale de ningún lado.

- **El cálculo de stock ya NO consulta por día.** `contexto_lote()` en
  `stock_producto.py` trae lote, dietas y movimientos una vez, y
  `calcular_consumo_diario_kg(..., ctx=...)` los reusa. Si tocás ese
  camino, pasá el `ctx`: sin él vuelve a pedirle todo a la base en cada
  día simulado. Medido: 4.116 consultas → 26 (99,4% menos), resultados
  idénticos. La fórmula de cantidad vigente tiene UNA sola
  implementación (`cantidad_vigente_desde_contexto`), y la versión que
  consulta la base la llama a ella, así que no pueden divergir.

- **Streamlit Cloud NO está auto-deployando.** La app quedó apuntando al
  nombre viejo del repo (`determinacion-de-pesos-y-conteo-bovinos-...`)
  y desde el rename del 29/07 no toma los commits nuevos. Un *Reboot app*
  desde share.streamlit.io sí baja el código (el pull funciona porque
  GitHub redirige), pero hay que hacerlo a mano después de cada push.
  Ojo: esto solo afecta la pantalla. Los crons de GitHub Actions hacen
  checkout en cada corrida, así que las alertas siempre corren el código
  nuevo.
- **Formularios en páginas pesadas.** Cada widget suelto dispara un rerun
  completo, y la ficha del cliente recalcula el stock de todos los lotes
  proyectando día por día. Los campos que se llenan de a varios van
  dentro de `st.form` o la pantalla se congela entre campo y campo.

- **Cualquier `.yml` dentro de `.github/workflows/` se ejecuta**, tenga
  el nombre que tenga. El guión bajo no lo exceptúa. `_env.yml` era solo
  comentarios y fallaba en cada push, dejando Actions en rojo de forma
  permanente — lo cual entrena a ignorar las fallas, justo cuando el
  backup depende de que se noten. Movido a `docs/variables-de-entorno.md`
  el 30/07/2026.

- **CUIT en clientes.** `normalizar_cuit` (deja solo dígitos),
  `cuit_valido` (largo **y** dígito verificador módulo 11),
  `formatear_cuit` y `buscar_cliente_por_cuit`, en `src/database.py`. Se
  guarda normalizado, así que da igual si se carga con guiones. La
  validación del dígito verificador no es adorno: en el sistema de
  facturación hay un cliente con `554445` en el campo CUIT, y cruzar por
  CUIT con datos así une clientes que no corresponden.
- **`_ensure_columna(tabla, columna, tipo)`** agrega columnas de forma
  perezosa y portable (`information_schema` en Postgres,
  `PRAGMA table_info` en SQLite). Es la vía para migrar la nube, porque
  `init_db` sale temprano cuando la base es Postgres.

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

### 31/07/2026 — El consumo no estaba mal cargado: estaba leyendo el campo equivocado

Venía a corregir los 5 valores de consumo que Mauricio validó a campo.
No hacía falta corregir ninguno: los 5 salen de una cuenta que usa el
campo equivocado de la dieta.

  - **Diagnóstico primero.** `scripts/../diag_consumo.py` (read-only,
    corre en la Mac porque el sandbox de la nube no tiene salida a
    Supabase) reconstruye por lote los dos caminos de cálculo que hoy
    conviven: el del stock (`DMI × pct_ms / 100`, materia seca) y el de
    la carga de silo (`kg_tal_cual × factor peso vivo`, tal cual). La
    columna que resolvió todo fue la **materia seca implícita**.

| Cliente | Producto | Sistema | `kg_tal_cual` | Campo | MS impl |
|---|---|---|---|---|---|
| Ezequiel Pezzola | Fibroter | 0,868 | **0,400** | 1,10 | **217%** |
| Jackie Graves | Fibrogreen Plus | 1,064 | 0,600 | 0,62 | 177% |
| Mario Salvadori | Fibrogreen Plus | 1,332 | 1,200 | 1,28 | 111% |
| Miguel Bergondi | Fibrogreen | 0,795 | 0,900 | 0,98 | 88% |
| Pedro M. Pezzola | Fibrogreen Plus | 0,786 | 0,700 | 0,75 | 112% |

  - La columna "Sistema" reproduce exactamente los 5 valores anotados
    el 30/07 (0,86 / 1,06 / 1,33 / 0,80 / 0,79), así que la
    reconstrucción es fiel. **`kg_tal_cual` le pega a la realidad con
    3-8% de error; la fórmula vieja erraba entre −21% y +72%.**
  - **Cambiado `calcular_consumo_diario_kg`** para que use
    `kg_tal_cual × cantidad`. Se conservan todas las claves del dict
    que devolvía (nadie fuera de `stock_producto.py` lee
    `pct_inclusion` ni `dmi_kg_animal`, pero se dejan por las dudas) y
    se agregan `kg_tal_cual_animal`, `fuente_kg` y `ms_implicita_pct`.
  - **Tres decisiones de borde, por si se revisan después:**
    1. Dieta vieja sin `kg_tal_cual` → cae a la fórmula anterior, pero
       `fuente_kg` queda marcado como "estimado" para que la pantalla
       lo pueda distinguir. Sin `kg_tal_cual` **y** sin DMI devuelve
       None: no se inventa un número.
    2. El `dmi_kg_dia_override` (ajuste por clima) ya no reemplaza el
       consumo, lo **escala** por `override / DMI_de_la_dieta`. Es la
       intención original: comen un x% más o menos que lo formulado.
    3. Un ingrediente con `pct_ms = 0` pero `kg_tal_cual` cargado (el
       "Rollo a voluntad" de Mario y Pedro) antes devolvía None porque
       la función salía por `pct <= 0`. Ahora se puede seguir. Es
       inocuo para las alertas: sin entregas cargadas el diagnóstico
       da `sin_entregas`, que `clientes_con_stock_bajo` ya excluye.
  - **Verificado:** `scripts/test_consumo_tal_cual.py`, 30 casos contra
    SQLite temporal, todos OK. Cubre los 5 lotes reales, plan de
    adaptación de 3 fases, bajas del lote, override de clima, dieta sin
    `kg_tal_cual`, producto ausente, y una prueba punta a punta de
    stock donde la fórmula vieja prometía 14 días y la nueva 8.
    `py_compile` OK y `nombres_huerfanos.py` sin huérfanos.
  - **Lo que el cambio NO arregla:** el `kg_tal_cual` de Fibroter en
    Ezequiel (0,400) es un dato roto y sigue roto — el test lo deja
    marcado a propósito en vez de taparlo. Y falta el escalado por peso
    vivo, que es el 3-8% restante (ver "Cosas que conviene saber").
**Segunda parte — la pesada del drone apagaba el escalado por peso.**
Mauricio marcó que la ganancia varía con el clima y que ese dato lo
tienen. Buscando dónde entra, apareció un bug peor.

  - `estimar_peso_vivo_lote` prioriza bien: si hay pesada real, la usa
    antes que la proyección. Pero **la devolvía congelada**, sin
    proyectar hacia adelante desde la fecha de la pesada. Como
    `factor_escala_consumo_pv` hace `peso_hoy / peso_a_la_fecha_de_la_
    dieta` y las dos fechas caían sobre la MISMA última pesada, el
    factor daba **1,000 exacto**. Un lote con pesada del drone perdía
    el ajuste por peso y uno sin datos sí lo recibía: al revés de lo
    que corresponde, y justo en los lotes mejor medidos.
  - **Arreglado:** ahora se proyecta desde la última pesada. La
    ganancia con la que se proyecta sale, si se puede medir, **de las
    dos últimas pesadas** — eso es lo que realmente pasó, e incorpora
    sin modelarlo el efecto del clima, la sanidad y la calidad del
    forraje. Si no se puede medir, cae al `adpv_objetivo_kg`.
  - Dos guardas para no comerse el ruido del drone: se exige un mínimo
    de 15 días entre pesadas (`_DIAS_MIN_ENTRE_PESADAS`) y la ganancia
    medida tiene que caer entre −0,5 y 3,0 kg/día. Fuera de eso, cae al
    objetivo. La pérdida de peso real sí se respeta: es dato, no error.
  - **Verificado:** `scripts/test_peso_proyectado.py`, 22 casos, todos
    OK. Cubre el bug original, ganancia medida vs objetivo, pesadas
    demasiado juntas, ganancia absurda, pérdida de peso, lote sin
    pesadas (no cambia nada), pesada del mismo día, pesada futura, el
    techo de 1,40 y la carga de silo punta a punta. Los 30 casos de
    `test_consumo_tal_cual.py` siguen pasando.
  - **Ojo para la próxima:** los factores 1,17-1,27 que salieron del
    diagnóstico usan el camino del `adpv_objetivo_kg`. En los lotes que
    tengan pesadas cargadas, el número real puede ser bastante menor.
    Antes de enchufar el escalado en el camino del stock conviene
    correr un diagnóstico de pesadas reales contra peso proyectado.

**Pusheado y desplegado.** Commit `73a629e`, push por GitHub Desktop y
reboot manual en share.streamlit.io (el auto-deploy sigue roto). La app
levantó bien con el cálculo nuevo.

**Tercera parte — el gráfico de evolución del peso (SIN PUSHEAR).**

  - `src/serie_peso.py` (nuevo, 43 casos de test en
    `scripts/test_serie_peso.py`) arma tres lecturas del mismo lote:
    proyección por ADPV objetivo, proyección ajustada por clima
    descontando lo que ya calculó `impacto_productivo`, y las pesadas
    reales de balanza y drone.
  - **La de clima es una BANDA, no una línea:** el impacto se estima en
    rango mín-máx y una línea sola aparentaría precisión que no hay.
  - **Eventos superpuestos: máximo, no suma** (dos registros del mismo
    frente frío describen el mismo evento). Un `confirmado` le gana a
    un `proyectado` del mismo día aunque su pérdida sea menor.
  - **La pérdida nunca hace bajar la curva:** el frío frena el engorde;
    que el animal pierda peso lo tiene que decir una pesada.
  - **`cobertura_clima` es la pieza importante:** qué % de los días del
    período tiene impacto registrado. Sin eso el gráfico miente por
    omisión — con 2 eventos en 60 días la curva de clima se pega a la
    del objetivo por FALTA DE DATOS, no porque no haya hecho frío, y
    quien mira concluye lo contrario. Abajo de 25% `resumen_desvio` lo
    dice con todas las letras.
  - En `app.py` reemplaza el gráfico de matplotlib de "Evolución de
    peso promedio" (que solo dibujaba las pesadas) por uno de altair
    con las tres series, y agrega barras de ADG por tramo abajo de la
    tabla de ADG. La app no usa plotly: `st.line_chart` y altair.
  - Paleta validada con el validador del skill de dataviz. **El verde
    de la casa quedó afuera del gráfico a propósito: verde y naranja
    no se distinguen en protanopía** (falla la separación CVD). Los
    cuatro colores son azul `#2a78d6`, naranja `#eb6834`, aqua
    `#1baf7a` y violeta `#4a3aa7`.
  - **Falta verlo andando:** el gráfico nunca se renderizó, porque el
    entorno donde se escribió no podía instalar altair. Antes de darlo
    por bueno hay que abrir la ficha de un lote y mirarlo.

**Cuarta parte — el incidente: los tests escribieron en producción.**

  - Al abrir la app para verificar el deploy, la lista de clientes
    mostraba "Absurda SA", "Drone SA", "EndToEnd SA"… y el header decía
    **33 clientes / 964 animales** en vez de 5 clientes. Los tests de
    la mañana habían corrido contra Supabase, no contra SQLite.
  - Causa y prevención: ver el primer punto de "Cosas que conviene
    saber". Los tres tests ahora pasan por `scripts/_sandbox_db.py`.
  - **Nada de los datos reales se tocó.** Los tests solo crean clientes
    nuevos y les cuelgan lotes propios. Ninguno tenía email ni WhatsApp,
    así que tampoco salió comunicación a nadie.
  - Limpieza hecha con `scripts/limpiar_datos_de_test.py`: 28 clientes
    (ids 8 al 35, todos con alta 2026-07-31) borrados; los 5 reales
    (ids 3 al 7, altas de mayo) protegidos por fecha y listados como
    tales antes de tocar nada. Quedaron 5 clientes.
  - El primer intento del script murió a mitad de camino por timeout de
    conexión: pedía los conteos de a uno por cliente, unas 150 idas y
    vueltas a Oregón. Reescrito para pedir todo de una y con reintentos.
    **Es el mismo problema que motiva el pendiente 485 (mudar la base a
    São Paulo)**, ahora con evidencia de que rompe cosas, no solo de que
    molesta.

### 30/07/2026 (noche) — El consumo por animal está mal cargado, y además NO debería ser fijo

Mauricio validó, uno por uno, los kg de producto por animal por día que
el sistema está usando. Dos están mal para el lado peligroso.

| Cliente | Sistema | Real (Mauricio) | Desvío |
| --- | --- | --- | --- |
| Ezequiel Pezzola | 0,86 | **1,10** | sistema 22% BAJO |
| Jackie Graves | 1,06 | **0,62** | sistema 71% ALTO |
| Mario Salvadori | 1,33 | **1,28** | 4%, aceptable |
| Miguel Bergondi | 0,80 | **0,98** | sistema 18% BAJO |
| Pedro Manuel Pezzola | 0,79 | **0,75** | 5%, aceptable |

"Bajo" es el lado peligroso: el sistema cree que comen menos de lo que
comen, promete más días de los que hay, y avisa tarde. La realidad lo
confirmó el mismo día: decía 302 kg de Fibroter en lo de Ezequiel y en
el campo había 90 (3 bolsas). Con el valor correcto, los 540 kg de hoy
no son 125 días sino ~98.

**Y el hallazgo de fondo, dicho por Mauricio: "por ahí tomás la
evolución del peso".** El consumo hoy es un número FIJO. No lo es: el
animal engorda, su consumo de materia seca sube, y el consumo del
producto sube con él. Ningún valor único puede ser correcto — sirve el
primer día y se desvía solo a partir de ahí. Los porcentajes de
inclusión además varían por producto, categoría, insumos y objetivo
(recría vs engorde).

Esto conecta el dron con la gestión: el dron mide pesos, los pesos deben
mover la ración, la ración el consumo, y el consumo el stock y las
alertas. Hoy esa cadena está cortada — se mide peso por un lado y se
calcula stock con un número congelado por el otro. Arreglar esto es lo
que convierte tres apps sueltas en un sistema.

**Pedido de Mauricio (30/07 noche), con fecha límite LUNES:**
  1. Corregir los errores de cálculo (los 5 valores + consumo por peso).
  2. Más rápido y ágil.
  3. Alertas a clientes: falta de producto por email Y WhatsApp; la
     climática ya funciona; y una nueva de cambio de ración con las
     indicaciones.
  4. Herramienta de análisis de precios de la competencia.
  5. Herramienta de negocio de engorde: hasta qué precio comprar y
     vender (se habló por chat en el celular, falta recuperar ese hilo).

### 30/07/2026 (tarde 4) — Diagnóstico de lentitud y arreglo del cálculo

Mauricio pidió un sistema ágil y rápido. Medí antes de opinar.

  - **`app.py` tiene 16.075 líneas y 10.796 (67%) se re-ejecutan en cada
    click** — eso es el modelo de Streamlit, no es optimizable.
  - **Un cálculo de stock disparaba 800 a 1.500 consultas SQL.** A la
    escala de hoy (5 clientes), dibujar la pantalla eran 4.872 consultas.
    Causa: el consumo se acumula y se proyecta día por día, y por cada
    día se volvía a pedir lote, dietas y cantidad de animales.
  - **El dashboard rápido es rápido porque está viejo:** lee un blob
    precalculado, y el cron declara 5 minutos pero GitHub Actions no
    cumple (vi datos de 23, 36 y 86 minutos). El propio código ya lo
    admite: *"6 horas de tolerancia: GitHub Actions NO corre el cron"*.
    O sea, el sistema ofrecía rápido **o** actual, nunca las dos.
  - **Arreglado:** contexto traído una vez. **4.116 → 26 consultas
    (99,4% menos)**, de ~140 ms a ~3 ms por cálculo, con resultados
    idénticos en 8 escenarios (incluidos planes de adaptación de dos
    fases y lotes con bajas) y en 5 fechas de cantidad vigente.
  - **Pendiente de decidir con datos:** si el cron de precálculo sigue
    haciendo falta. Probablemente no, y sacarlo devolvería datos en vivo.

### 30/07/2026 (tarde 3) — Backup verificado en producción

  - **El backup corrió de verdad y anduvo**: `pg_dump` 17 contra Supabase
    a través del pooler, dump de 0,29 MB con 51 tablas con datos,
    verificación OK y mail enviado. 37 segundos en total. La duda que
    quedaba (si el pooler se llevaría bien con `pg_dump`) está resuelta.
  - `_env.yml` movido a `docs/variables-de-entorno.md`: fallaba en cada
    push y dejaba Actions siempre en rojo.
  - Verificado también: los 3 commits en `master`, el rol
    `hms_ganadera_ro` creado con permisos mínimos (solo `SELECT` sobre
    `comprobantes`, `clientes` y `productos`, sin superusuario), y la app
    en producción levantando bien después del deploy.

### 30/07/2026 (tarde 2) — CUIT en clientes

Primera pieza de la integración con facturación, y la única que no
depende de credenciales ni de nada externo.

  - Columna `cuit` en `clientes`, con `_ensure_columna` para que migre
    sola en las dos bases. Probado sobre una SQLite vieja sin la columna
    (migra al primer uso y no toca los clientes existentes) y la rama
    Postgres contra un Postgres local.
  - Validación con dígito verificador. Verificada contra los CUIT reales
    del sistema de facturación: los 7 válidos pasan, el `554445` de
    Moya se rechaza, y cambiarle un dígito al de Pezzola también.
  - Campo en el alta y en la edición de cliente, con aviso en vivo si el
    CUIT no cierra.

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
