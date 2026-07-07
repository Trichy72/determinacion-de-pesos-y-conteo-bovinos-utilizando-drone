"""
Diagnóstico exhaustivo: corre YOLO con varios umbrales y modelos sobre el
video, sin filtrar por clase, para ver QUÉ detecta y CÓMO clasifica los
objetos. Útil cuando el modelo de COCO no reconoce vacas desde arriba.

Uso:
    python scripts/diagnostico.py "/ruta/al/video.mp4"
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/diagnostico.py /ruta/al/video.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1]).expanduser().resolve()
    print(f"📹 Video: {video_path}")
    if not video_path.exists():
        print(f"❌ ARCHIVO NO ENCONTRADO: {video_path}")
        print("   Tip: arrastrá el video a la terminal para que se complete la ruta.")
        sys.exit(1)

    out_dir = Path("diagnostico_output")
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ OpenCV no pudo abrir el video.")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total == 0:
        print("❌ Video sin frames.")
        sys.exit(1)
    print(f"   {total} frames @ {fps:.1f} fps  ({total/fps:.1f} segundos)")

    # Frame del 80% — donde más vacas vimos
    fidx = int(total * 0.80)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("❌ No pude leer el frame de prueba.")
        sys.exit(1)

    print(f"\n🎯 Probando frame {fidx} (t={fidx/fps:.1f}s)")
    print(f"   Resolución: {frame.shape[1]}x{frame.shape[0]}")

    # Probaremos 3 modelos × 2 confianzas, todas las clases
    configs = [
        ("yolov8s.pt", 0.05),
        ("yolov8s.pt", 0.20),
        ("yolov8m.pt", 0.05),  # modelo más grande
        ("yolov8m.pt", 0.20),
    ]

    print("\n" + "=" * 70)
    print(f"{'MODELO':<15} {'CONF':<6} {'#TOTAL':<8} {'CLASES DETECTADAS'}")
    print("=" * 70)

    best_result = None
    best_count = 0

    for model_path, conf in configs:
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"❌ {model_path}: {e}")
            continue

        results = model.predict(frame, conf=conf, iou=0.5, imgsz=1280, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            print(f"{model_path:<15} {conf:<6} {'0':<8} (nada)")
            continue

        clases = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        # Resumen por clase
        from collections import Counter
        counter = Counter(int(c) for c in clases)
        names = model.names
        resumen = ", ".join(
            f"{names.get(c, c)}×{n}" for c, n in counter.most_common(5)
        )
        n_total = len(clases)
        print(f"{model_path:<15} {conf:<6} {n_total:<8} {resumen}")

        if n_total > best_count:
            best_count = n_total
            best_result = (model_path, conf, r, model.names)

    # Guardar la MEJOR detección anotada
    if best_result is not None:
        model_path, conf, r, names = best_result
        annotated = frame.copy()
        for box, cf, cls in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = map(int, box)
            name = names.get(int(cls), str(cls))
            color = (0, 255, 0) if cls == 19 else (0, 165, 255)  # verde si cow
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            cv2.putText(annotated, f"{name} {cf:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        out_file = out_dir / f"mejor_deteccion_{model_path}_{conf}.jpg"
        cv2.imwrite(str(out_file), annotated)
        print(f"\n💾 Mejor detección guardada en: {out_file}")
        print(f"   Modelo: {model_path}  Conf: {conf}  Total objetos: {best_count}")
    else:
        print("\n⚠️  Ningún modelo detectó nada en este frame.")
        print("   Esto significa que YOLO base de COCO no ve tus vacas.")
        print("   Solución: hay que hacer fine-tuning con dataset aéreo de bovinos.")

    print(f"\n✅ Listo. Mirá la carpeta: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
