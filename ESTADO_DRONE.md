# Estado del proyecto — Drone (conteo y peso de bovinos)

> **Para Claude:** este archivo es tu memoria de este frente de trabajo.
> Leelo completo antes de tocar nada. Al terminar la sesión, actualizalo:
> qué se hizo, qué decisiones se tomaron y por qué, y qué quedó pendiente.
> **Importante:** este proyecto ya tuvo dos intentos fallidos por saltear
> la verificación. Leé la sección "Errores cometidos" antes de proponer
> nada, para no repetirlos.

**Última actualización:** 29/07/2026

---

## Qué es

Sistema que a partir de fotos y videos de drone cuenta bovinos y estima su
peso, sin pasarlos por la balanza. Corre **local en la Mac** (necesita GPU
y archivos pesados); lo que calcula se guarda en la ficha del cliente de la
app online, que es el nexo entre los dos mundos.

- **App local:** `drone_app.py`. Se abre con el alias `drone` en Terminal,
  o doble click en `Drone HMS.command`.
- **Pipeline compartido:** `src/recorrida.py`
- **Config:** `config.yaml`

## Cómo funciona el cálculo (validado)

1. **Escala:** la altura de vuelo sale del XMP `RelativeAltitude` de la
   propia foto. GSD = alt × 100 × 36 / (f35 × ancho_px), con f35 = 24.
   Fotos 4000×2250, frames de video 3840×2160. **No se usa referencia en
   el piso** (la lona de 1,02 m y los marcadores ArUco quedaron obsoletos).
2. **Detección:** `yolov8l-seg.pt` (segmentación).
3. **Área:** se mide sobre la **máscara**, no el bounding box. Esto es
   crítico: en invierno con sol bajo la caja incluye la sombra e infla el
   peso ~30 %.
4. **Peso:** alométrico `peso = a × área^b`, con `coef_a = 250.4` y
   `coef_b = 1.20`. El coeficiente se calibró contra balanza (corrales 3 y
   5 de La Esperanza).
5. **Filtros:** se descartan los animales cortados por el borde (cuentan
   para el conteo pero no para el peso, porque se mide media silueta) y se
   usa media recortada al 10 % para el promedio.

## Precisión real (medida, no estimada)

Validado el 27/07/2026 contra balanza en La Esperanza Argentina:

| Corral | Real (balanza) | Sistema | Error |
|---|---|---|---|
| 3 | 460-500 kg (78 nov.) | — | **−6 %** |
| 5 | 440-460 kg (78 nov.) | 465 kg | **+3 %** |
| 6 | 380-440 kg (96 anim.) | — | +22 % → descartado |

- **Los pesos de balanza son peso LLENO**, tomados el día anterior a la
  filmación.
- El corral 6 dio +22 % porque solo había fotos a 50-100 m. De ahí la regla
  siguiente.

## Reglas duras (no negociar)

- **Peso solo con fotos de 10 a 20 m.** Arriba de 20 m el animal ocupa
  pocos píxeles y el peso se va para arriba. El sistema avisa cuando las
  fotos no son confiables.
- **La caja/máscara cubre el cuerpo, nunca la sombra.**
- **El conteo es más difícil que el peso.** En tropas apretadas el detector
  genérico pierde animales (contó 2 de 3 en un video). Ese es justamente el
  problema que el fine-tuning intenta resolver.

## Dónde está trabado: el fine-tuning

**Objetivo:** entrenar un detector propio (`hms_bovinos.pt`) que reconozca
bovinos desde arriba, porque el modelo genérico (COCO) nunca vio esa vista.

**Estado:** dos intentos fallidos, tercero a medio camino.

### Errores cometidos (leer antes de proponer algo)

1. **Etiquetas estimadas por un modelo de visión → mAP50 = 0.018.**
   Le pedí a agentes con visión que estimaran las coordenadas de cada caja
   "a ojo". Las cajas quedaron **desalineadas**: caían sobre tierra pelada
   y dejaban animales sin marcar. Un modelo de lenguaje describe bien lo
   que ve pero **no acierta coordenadas de píxel**. El entrenamiento no
   aprendió nada.
   → **Lección:** las cajas las pone un detector, no un LLM. Lo que un LLM
   sí puede hacer es **juzgar si a una imagen le faltan animales** (eso es
   percepción global, no coordenadas). Usarlo para control de calidad, no
   para etiquetar.

2. **Verificar DESPUÉS de entrenar es tarde.** Se gastó una corrida de
   Colab entera para descubrir que el dataset estaba mal. Ahora se dibuja
   las cajas sobre las fotos y se mira ANTES de subir nada.

### Lo que sí funcionó

**Detección por mosaicos** (`src/mosaico.py`): parte la foto en cuadrados
de 1024 px con 30 % de solapamiento y detecta en cada uno, así el animal le
llega al detector a resolución casi nativa en lugar de reducido 3×.
Descarta las cajas cortadas por el borde del mosaico (el vecino las tiene
enteras) y hace NMS global.

Resultado: las cajas quedan **bien ubicadas**. Cobertura ~60 % (en
DJI_0084: 35 detectados de ~55-60 reales).

Mejor combinación medida sobre 3 fotos (`scripts/probar_deteccion.py`):
`conf 0.05`, `tile 1024`, `overlap 0.30`.

### Los tres caminos evaluados

- **Vía A — dataset público.** Existen datasets de bovinos con drone
  etiquetados por investigadores: OpenCows2020 (3.707 imágenes, 6.917
  vacas, descarga directa de 2,1 GB con labels YOLO), Cattle Detection and
  Counting in UAV Images (670 img). **Dos reparos:** son Holstein blanco y
  negro en pastura, no Angus/Hereford sobre tierra, así que cuánto
  transfiere hay que medirlo; y la licencia de OpenCows2020 es **NO
  comercial**, y HMS es actividad comercial → habría que buscar uno con
  licencia CC BY (Roboflow Universe) o pedir permiso a los autores.
- **Vía B — ELEGIDA. Entrenamiento en dos vueltas.** Entrenar solo con las
  imágenes donde el pre-etiquetado ya está casi completo (fotos a 10 m con
  animales separados). Con ese modelo, re-etiquetar todo automáticamente y
  hacer una segunda vuelta. Cero trabajo manual.
- **Vía C — etiquetado manual.** 30 imágenes en makesense.ai, ~1 hora de
  Mauricio. Resultado seguro. Ya está preparado por si B no alcanza
  (`scripts/preparar_etiquetado_manual.py` deja las imágenes con el
  pre-etiquetado cargado y un INSTRUCCIONES.txt).

## Próximo paso concreto

**Resultado de la Vía B (29/07/2026): no alcanza sola.** Se corrió
`seleccionar_limpias.py` sobre las 100 imágenes y se revisaron las
candidatas mejor puntuadas. Ni esas están completas:

- #1 DJI_0089_t009 (puntaje 0.76): 7 marcados de ~13 reales.
- #4 DJI_0082 (0.62): 6 de ~9, y una caja cae sobre una **sombra**.
- #5 DJI_0120_t021 (0.62): 4 cajas perfectas, falta **1 animal**.

El puntaje **ordena bien** (las manadas apretadas quedan abajo) pero el
nivel absoluto de cobertura no llega a un dataset limpio.

**Lo que sí cambió:** el trabajo manual sobre estas 40 candidatas es
chico. Son escenas fáciles donde falta agregar **1 a 3 cajas por imagen**,
no 20. Estimación: 20-30 minutos, no una hora.

**Por eso el próximo paso es la Vía C sobre las 40 candidatas:**

```bash
python scripts/preparar_etiquetado_manual.py --n 40
```

El script ahora detecta `candidatas_limpias/ranking.csv` y usa ese orden
(las que menos trabajo necesitan) en lugar de elegir por variedad. Deja las
imágenes + labels + INSTRUCCIONES.txt en `etiquetado_manual/`, listas para
makesense.ai. Arranca con 188 cajas ya puestas (4,7 por imagen).

Después de exportar el zip de makesense: armar el dataset, entrenar en
Colab, y con el modelo resultante re-etiquetar las 100 imágenes
automáticamente (segunda vuelta).

## Datos de campo disponibles

Recorrida del viernes 24/07/2026, dos clientes:

- **Roxdan:** DJI_0063-0078. Sin datos de balanza.
- **La Esperanza Argentina:**
  - Corral 1: DJI_0079-0084 — 185 terneras, sin peso
  - Corral 3: DJI_0086-0116 — 78 novillos de 460 a 500 kg
  - Corral 5: DJI_0117-0129 — 78 novillos de 440 a 460 kg
  - Corral 6: DJI_0132-0148 — 96 animales de 380 a 440 kg

Archivos en `videos_drone/tarjeta_drone` (5,8 GB, fuera de git).

## Scripts

| Script | Para qué |
|---|---|
| `preparar_dataset.py` | Selecciona fotos ≤20 m + frames de video, pre-etiqueta con mosaicos, arma estructura YOLO + zip |
| `probar_deteccion.py` | Compara combinaciones de conf/tile/overlap sobre pocas fotos, para elegir antes de procesar todo |
| `seleccionar_limpias.py` | Puntúa qué imágenes tienen el pre-etiquetado completo |
| `preparar_etiquetado_manual.py` | Prepara 30 imágenes para completar a mano en makesense.ai (Vía C) |
| `procesar_recorrida.py` | Procesa una recorrida completa y calibra contra datos reales de balanza |

## Pendientes

- Terminar el fine-tuning (ver "Próximo paso").
- Probar el modelo nuevo contra el video donde contaba 2 de 3 animales.
- Crear Roxdan y La Esperanza como clientes en la app online, para poder
  guardar las recorridas en ficha.
- Medición de comederos y bebederos (mencionado, sin empezar).
- Peso desde video: necesita la altura de grabación, que está en la tarjeta
  del comando (grabación de pantalla con las métricas).
- Toggle de desbaste (para comparar contra pesos vacíos).

## Cosas que conviene saber antes de tocar

- **El sandbox de Claude no puede correr YOLO** (pip bloqueado por el
  proxy). Todo lo que necesite ultralytics lo corre Mauricio en la Mac, que
  tiene el venv `.venv` con todo instalado.
- **Git en esta carpeta:** mount FUSE, `unlink` prohibido. Ver
  `ESTADO_GESTION.md`, sección "Cosas que conviene saber".
- **Colab:** el notebook lee el zip desde Google Drive, no por el botón de
  subir. El botón de Colab se corta si la Mac se duerme o pasa un par de
  minutos sin actividad; Drive retoma solo.
- **Las imágenes del dataset se redimensionan a 1920 px** antes de armar el
  zip: el entrenamiento va a 1280 igual y el zip baja de 305 MB a 64 MB.
