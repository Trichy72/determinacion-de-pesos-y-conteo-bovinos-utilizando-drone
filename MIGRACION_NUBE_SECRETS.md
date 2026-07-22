# Cargar secrets en GitHub Actions y Streamlit Cloud

Este documento tiene el paso a paso para cargar las credenciales que
hoy viven en `data/*.json` de la Mac, en los dos entornos de nube.
Sin esto, los crons de GitHub Actions y la app cloud no pueden mandar
emails, WhatsApp ni consultar clima ni la IA.

## 1. Sacar los valores de la Mac

En Terminal de la Mac:

```bash
cd "/Users/hms/Documents/Claude/Projects/determinacion de pesos y conteo bovinos utilizando drone"

# SMTP + IMAP
cat data/smtp_config.json

# Twilio WhatsApp
cat data/whatsapp_config.json

# WeatherAPI
cat data/weatherapi_config.json

# Anthropic + DATABASE_URL
cat .env
```

Anotá cada valor. Los vas a pegar en dos lados.

## 2. GitHub → Settings → Secrets → Actions

1. Andá a `https://github.com/Trichy72/determinacion-de-pesos-y-conteo-bovinos-utilizando-drone/settings/secrets/actions`
2. Click **New repository secret** por cada línea de la tabla.
3. Nombre exacto (mayúsculas) y valor sin comillas.

| Secret name              | De dónde sale                     | Ejemplo             |
| ------------------------ | --------------------------------- | ------------------- |
| `DATABASE_URL`           | `.env` línea DATABASE_URL         | postgresql://...    |
| `ANTHROPIC_API_KEY`      | `.env` o data/.api_key            | sk-ant-…            |
| `SMTP_HOST`              | smtp_config.json → host           | smtp.gmail.com      |
| `SMTP_PORT`              | smtp_config.json → port           | 465                 |
| `SMTP_USER`              | smtp_config.json → user           | hms002@gmail.com    |
| `SMTP_PASSWORD`          | smtp_config.json → password       | app password        |
| `SMTP_FROM_EMAIL`        | smtp_config.json → from_email     | hms002@gmail.com    |
| `SMTP_FROM_NAME`         | smtp_config.json → from_name      | HMS Nutrición Animal|
| `SMTP_USE_SSL`           | smtp_config.json → use_ssl        | true                |
| `SMTP_USE_TLS`           | smtp_config.json → use_tls        | false               |
| `SMTP_ADMIN_EMAIL`       | smtp_config.json → admin_email    | hms002@gmail.com    |
| `SMTP_BCC_CLIENTES`      | smtp_config.json → bcc_clientes   | true                |
| `IMAP_HOST`              | smtp_config.json → imap_host      | imap.gmail.com      |
| `IMAP_USER`              | smtp_config.json → imap_user      | hms002@gmail.com    |
| `IMAP_PASSWORD`          | smtp_config.json → imap_password  | app password        |
| `TWILIO_ACCOUNT_SID`     | whatsapp_config.json → account_sid| AC…                 |
| `TWILIO_AUTH_TOKEN`      | whatsapp_config.json → auth_token | …                   |
| `TWILIO_FROM_NUMBER`     | whatsapp_config.json → from_number| +14155238886        |
| `TWILIO_ADMIN_PHONE`     | whatsapp_config.json → admin_phone| +5492954517407      |
| `TWILIO_MODO_SANDBOX`    | whatsapp_config.json → modo_sandbox| true               |
| `CARGA_BASE_URL`         | whatsapp_config.json → carga_base_url | https://…       |
| `WEATHERAPI_KEY`         | weatherapi_config.json → api_key  | …                   |

## 3. Variable de dry-run (para la fase de prueba)

En `Settings → Secrets → Variables` (pestaña Variables, no Secrets)
crear una variable llamada `DRY_RUN` con valor `true`. Con eso los
scripts corren pero no envían emails ni WhatsApp reales. Cuando pases
a producción, cambiala a `false`.

## 4. Streamlit Cloud → App settings → Secrets

En `https://share.streamlit.io/`, entrá a tu app, **Settings →
Secrets**. Se cargan en formato TOML — pegá exactamente esto (con tus
valores):

```toml
DATABASE_URL = "postgresql://..."
ANTHROPIC_API_KEY = "sk-ant-..."

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "465"
SMTP_USER = "hms002@gmail.com"
SMTP_PASSWORD = "..."
SMTP_FROM_EMAIL = "hms002@gmail.com"
SMTP_FROM_NAME = "HMS Nutrición Animal"
SMTP_USE_SSL = "true"
SMTP_USE_TLS = "false"
SMTP_ADMIN_EMAIL = "hms002@gmail.com"
SMTP_BCC_CLIENTES = "true"

IMAP_HOST = "imap.gmail.com"
IMAP_USER = "hms002@gmail.com"
IMAP_PASSWORD = "..."

TWILIO_ACCOUNT_SID = "AC..."
TWILIO_AUTH_TOKEN = "..."
TWILIO_FROM_NUMBER = "+14155238886"
TWILIO_ADMIN_PHONE = "+5492954517407"
TWILIO_MODO_SANDBOX = "true"
CARGA_BASE_URL = "https://..."

WEATHERAPI_KEY = "..."
```

Streamlit los expone como `st.secrets["..."]`, pero también como
variables de entorno normales `os.getenv(...)`, así que el código
funciona igual que en GitHub Actions.

## 5. Verificar que todo esté cargado

En GitHub Actions:

1. Andá a la pestaña **Actions**.
2. Elegí un workflow (ej. **Alertas diarias**).
3. Click **Run workflow → Run workflow**.
4. Con `DRY_RUN=true` no manda nada real, pero verás en el log si
   los secrets están bien.

En Streamlit Cloud: abrí la app, andá a la pestaña **Configuración**,
y clickeá "Enviar email de prueba". Si tenés SMTP mal cargado, ahí
lo vas a ver.

## 6. Rotación de credenciales

Si cambiás una contraseña (Gmail app password, Twilio auth token,
etc.), acordate de:

1. Actualizar el JSON local en la Mac.
2. Actualizar el secret en GitHub (Settings → Secrets → editar).
3. Actualizar la línea del TOML en Streamlit Cloud.

Los 3 lugares tienen que estar sincronizados hasta que apaguemos la
Mac. Después de eso, solo GitHub + Streamlit.
