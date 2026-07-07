# 🐄 Conteo y estimación de peso bovino por drone

App web para cargar **imágenes** o **videos** capturados con drone (4K @ 30 fps,
~10 m de altura, vista cenital con cuadrado de referencia de 1,02 m) y obtener:

- ✅ Conteo de animales del lote
- ✅ Peso estimado individual y promedio (objetivo: <5 % de error)
- ✅ Video o imagen anotada con cajas + ID + peso por animal
- ✅ Tabla CSV exportable de resultados

## 🚀 Instalación

```bash
# 1) Python 3.10+ recomendado
python -m venv .venv
source .venv/bin/activate     # en Windows: .venv\Scripts\activate

# 2) Dependencias
pip install -r requirements.txt

# 3) (opcional) ffmpeg para mejor codificación de video
#    macOS:   brew install ffmpeg
#    Ubuntu:  sudo apt install ffmpeg
#    Windows: choco install ffmpeg
```

La primera vez que se ejecute, `ultralytics` descargará automáticamente
los pesos de YOLOv8.

## ▶️ Cómo usarla

```bash
streamlit run app.py
```

Se abre `http://localhost:8501`. Subí imagen o video, ajustá la raza y
la altura de vuelo en la barra lateral, y listo.

## 📐 Captura recomendada

| Parámetro | Valor |
|-----------|-------|
| Resolución | 4K (3840×2160) |
| Frame rate | 30 fps |
| Altura | ~10 m |
| Ángulo cámara | 90° (cenital) |
| Referencia piso | Cuadrado de 1,02 × 1,02 m (ArUco o color sólido) |

> 💡 **Tip**: imprimí un marcador ArUco DICT_4X4_50 ID 0 a 1,02 m × 1,02 m
> sobre tela vinílica. Es mucho más robusto que un cuadrado de color.

## 🎯 Calibración para alcanzar <5 % de error

El modelo viene con coeficientes razonables de literatura, pero **para tu
rodeo y condiciones conviene calibrarlos** con tus propios datos.

1. Tomá ~30-50 imágenes donde aparezca **un solo animal** y la referencia
   de 1,02 m visible.
2. Pesá cada animal en balanza.
3. Llená un CSV (ver `data/calibracion_template.csv`):

   ```csv
   image_path,peso_kg,raza
   data/calibracion/img001.jpg,420.5,angus
   data/calibracion/img002.jpg,380.0,hereford
   ...
   ```

4. Corré el calibrador:

   ```bash
   python scripts/calibrate_weight.py \
       --dataset data/calibracion.csv \
       --output models/weight_model.json \
       --yolo yolov8m-seg.pt
   ```

5. En la app, en la barra lateral, activá **“Usar modelo de peso calibrado”**
   y subí `weight_model.json`.

## 🧠 Cómo funciona

```
imagen/video
    ↓
[Calibración]   detección de cuadrado 1,02 m  →  px/metro
    ↓
[Detección]     YOLOv8 (clase 'cow', COCO id 19)
    ↓
[Tracking]      ByteTrack → ID estable por animal (sólo video)
    ↓
[Área]          píxeles de bbox/máscara × (m/px)²  =  área m²
    ↓
[Peso]          Peso(kg) = a · Area^b · factor_raza + c
    ↓
[Salida]        conteo, peso individual y promedio, video anotado, CSV
```

### Modelo de peso

Por defecto usa una ley alométrica:

> **Peso (kg) = 220 · Área^1.20 · factor_raza**

donde `factor_raza` es 1.00 para Angus, 0.97 para Hereford, 1.05 para
Brangus y 1.03 para Braford. Estos valores son de literatura veterinaria
y deben calibrarse con datos reales para precisión <5 %.

## 📁 Estructura

```
.
├── app.py                       # UI Streamlit
├── config.yaml                  # Parámetros del sistema
├── requirements.txt
├── README.md
├── src/
│   ├── calibration.py           # ArUco / cuadrado color
│   ├── detector.py              # YOLOv8 wrapper
│   ├── weight_estimator.py      # Modelo Peso=f(Area)
│   └── processor.py             # Pipeline imagen + video
├── scripts/
│   └── calibrate_weight.py      # Calibrador de coeficientes
└── data/
    └── calibracion_template.csv
```

## 🛣️ Roadmap

- [ ] Fine-tuning de YOLO con dataset de bovinos cenitales
- [ ] Modelo separado por clase de edad (terneros / vacas / toros)
- [ ] Integración con DJI SDK para captura en vivo
- [ ] Despliegue en cloud (Docker + GPU)
- [ ] Modo offline / mobile (TFLite)

## 📚 Referencias

- Tasdemir et al. (2011) – *Determination of body measurements on the
  Holstein cows using digital image analysis*
- Cominotte et al. (2020) – *Automated computer vision system to predict
  body weight and average daily gain in beef cattle*
- Huxley (1932) – *Problems of relative growth* (ley alométrica)
