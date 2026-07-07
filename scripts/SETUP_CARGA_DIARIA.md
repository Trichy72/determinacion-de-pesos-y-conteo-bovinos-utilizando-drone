# Setup — Carga diaria por WhatsApp

Sistema que pide al encargado de cada lote, todos los días a las
17:00, cuántos kg cargó de cada ingrediente al comedero. El encargado
recibe un WhatsApp con un link tokenizado a un mini-form web; cuando
lo envía, queda registrado como carga del día en la app principal.

## Componentes

1. **`cargar_diaria_app.py`** — mini-app Streamlit independiente con
   el form que llena el encargado.
2. **`scripts/whatsapp_pedido_carga.py`** — cron diario que recorre
   lotes activos y manda WhatsApp con el link firmado.
3. **`scripts/com.hms.pedido-carga.plist`** — launchd que dispara el
   cron a las 17:00.
4. **`src/carga_diaria_token.py`** — generador/validador de tokens
   HMAC para que el link no se pueda falsificar.

## Configuración inicial (una sola vez)

### 1) Levantar la mini-app en su propio puerto

```bash
cd "/Users/hms/Documents/Claude/Projects/determinacion de pesos y conteo bovinos utilizando drone"
.venv/bin/streamlit run cargar_diaria_app.py --server.port 8502
```

Dejala corriendo en una terminal o configurá un launchd separado para
arrancarla con tu Mac.

### 2) Exponer el puerto con un túnel público

Tu Mac no tiene IP pública, entonces necesitás un túnel que le dé
una URL HTTPS a `localhost:8502`. Dos opciones gratuitas y simples:

**Opción A — ngrok (la más fácil para empezar)**

```bash
# Instalar (una sola vez)
brew install ngrok

# Login (una sola vez): tomá el token de https://dashboard.ngrok.com
ngrok config add-authtoken TU_TOKEN

# Levantar el túnel (cada vez que querés que el sistema funcione)
ngrok http 8502
```

Te va a dar una URL tipo `https://abc123.ngrok-free.app`. Copiala.

> ⚠️ La URL gratuita de ngrok **cambia cada vez** que reiniciás el
> túnel. Si querés URL fija, pagás un plan ngrok ($8/mes) o usá
> Cloudflare Tunnel (gratis y fijo).

**Opción B — Cloudflare Tunnel (URL fija y gratis)**

```bash
brew install cloudflared
cloudflared tunnel login
cloudflared tunnel create hms-carga
cloudflared tunnel route dns hms-carga carga.tudominio.com
cloudflared tunnel run --url http://localhost:8502 hms-carga
```

Necesitás un dominio en Cloudflare (gratis si registrás uno).

### 3) Cargar la URL base en la app

Editá `data/whatsapp_config.json` y agregá la clave `carga_base_url`:

```json
{
  "twilio_account_sid": "...",
  "twilio_auth_token": "...",
  "twilio_from": "whatsapp:+14155238886",
  "carga_base_url": "https://abc123.ngrok-free.app"
}
```

(O hacelo desde la app: Configuración → WhatsApp → URL pública del
túnel, una vez que sumemos ese input.)

### 4) Configurar el encargado por lote

En la app principal:

1. Andá a **🏢 Clientes/Lotes** → seleccioná el lote.
2. Abrí el expander **📱 Encargado del lote — carga diaria por
   WhatsApp**.
3. Cargá nombre y WhatsApp del encargado.
4. Activá el toggle **"Activar pregunta diaria 17:00"**.
5. Probá el link de prueba que aparece debajo.

### 5) Programar el cron 17:00

```bash
cd "/Users/hms/Documents/Claude/Projects/determinacion de pesos y conteo bovinos utilizando drone"

# 1) Reemplazar la ruta placeholder en el plist
PROY="$PWD"
sed "s|REEMPLAZAR_RUTA_PROYECTO|$PROY|g" \
    scripts/com.hms.pedido-carga.plist \
    > ~/Library/LaunchAgents/com.hms.pedido-carga.plist

# 2) Cargar el launchd
launchctl load ~/Library/LaunchAgents/com.hms.pedido-carga.plist
```

Para probar antes del horario real:

```bash
# Dry-run (no envía, solo muestra qué pasaría)
.venv/bin/python3 scripts/whatsapp_pedido_carga.py --dry-run

# Envío real a un solo lote
.venv/bin/python3 scripts/whatsapp_pedido_carga.py --solo-lote 7
```

## Cómo se ve para el encargado

Recibe WhatsApp:

> Hola Juan, te paso el link para que cargues la dieta de hoy del
> lote *Recría hembras* (Jackie Graves).
>
> 📋 Recomendado hoy: ~287 kg de mezcla total.
>
> 👉 https://abc.ngrok-free.app/?token=7.20260525.d560a047
>
> El link tiene un form pre-cargado con los ingredientes — solo
> ajustá los kg si fue distinto y dale Enviar.
>
> Gracias!
> — HMS Nutrición Animal

Toca el link, abre el form web con los ingredientes pre-llenados,
ajusta si hace falta y aprieta **Confirmar y enviar al asesor**. La
carga queda automáticamente registrada como `lineal_diario` en la
DB, con desglose por ingrediente.

## Seguridad

- Token firmado con HMAC-SHA256 + clave secreta local (32 bytes).
- Cada token vale solo para `(lote_id, fecha)` y caduca a las 48 hs.
- Sin login: la URL ES la credencial. Quien tiene el link puede
  cargar. Mauricio comparte el link solo con el encargado correcto.
- La clave secreta se guarda en `data/.carga_secret` (permisos 600).
  Si la borrás, todos los tokens previos quedan inválidos.

## Troubleshooting

- **El encargado dice que el link "no anda"** → ¿el túnel está
  corriendo? ¿La URL del link coincide con la URL actual del túnel?
  Si usás ngrok gratuito, la URL cambia en cada reinicio.

- **Twilio sandbox: el mensaje no llega** → el encargado tiene que
  mandar `join <código>` al número sandbox de Twilio una sola vez
  antes de recibir mensajes. El código lo encontrás en la consola
  Twilio → Messaging → Try it out → Send a WhatsApp message.

- **"Token vencido"** → el link es del día anterior (o anterior).
  Pedile al sistema que mande uno nuevo (`--solo-lote N` manual).

- **"No hay dieta vigente"** → cargá una dieta en el lote (Asesor IA
  o pestaña Análisis) antes de activar la pregunta diaria.
