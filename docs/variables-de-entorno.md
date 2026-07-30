# Variables de entorno del sistema HMS

Este archivo vivía en `.github/workflows/_env.yml` con un comentario que
decía "este archivo NO es un workflow (empieza con `_`)". Esa suposición
era falsa: el guión bajo no hace que GitHub lo ignore. **Cualquier `.yml`
dentro de `.github/workflows/` se intenta ejecutar**, y como este era solo
comentarios, fallaba en cada push. Tres pushes seguidos del 30/07/2026
figuraban en rojo por eso.

El problema no era funcional, pero acostumbraba a ver Actions en rojo y a
ignorarlo — justo cuando el workflow de backup depende de que una falla
se note. Por eso se movió acá.

## Dónde se cargan

| Entorno | Dónde |
| --- | --- |
| GitHub Actions | Settings → Secrets and variables → Actions |
| Streamlit Cloud | App settings → Secrets (formato TOML) |
| Mac (desarrollo) | `.env` y `data/*.json` |

La lista canónica de los 22 secrets, con el formato exacto para pegar en
Streamlit, está en **`.streamlit/secrets.toml.example`**. El paso a paso
para cargarlos está en **`MIGRACION_NUBE_SECRETS.md`**.

## Las variables

### Base
| Variable | Ejemplo |
| --- | --- |
| `DATABASE_URL` | `postgresql://...` (session pooler de Supabase) |
| `ANTHROPIC_API_KEY` | `sk-ant-...` |

### SMTP (emails)
| Variable | Ejemplo |
| --- | --- |
| `SMTP_HOST` | `smtp-relay.brevo.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | usuario del relay |
| `SMTP_PASSWORD` | clave del relay |
| `SMTP_FROM_EMAIL` | `alertas@hmsnutricionanimal.com.ar` |
| `SMTP_FROM_NAME` | `HMS Nutrición Animal` |
| `SMTP_USE_SSL` | `false` |
| `SMTP_USE_TLS` | `true` |
| `SMTP_ADMIN_EMAIL` | casilla que recibe el digest y los backups |
| `SMTP_BCC_CLIENTES` | email que recibe copia oculta de cada mail a cliente |

### IMAP (procesador de bajas)
| Variable | Ejemplo |
| --- | --- |
| `IMAP_HOST` | `imap.mail.me.com` |
| `IMAP_USER` | casilla |
| `IMAP_PASSWORD` | clave de aplicación |

### WhatsApp (Twilio)
| Variable | Ejemplo |
| --- | --- |
| `TWILIO_ACCOUNT_SID` | `AC...` |
| `TWILIO_AUTH_TOKEN` | token |
| `TWILIO_FROM_NUMBER` | `+14155238886` (sandbox) — sin el prefijo `whatsapp:`, lo agrega el código |
| `TWILIO_ADMIN_PHONE` | teléfono del admin |
| `TWILIO_MODO_SANDBOX` | `true` o `false` |
| `CARGA_BASE_URL` | URL pública de la mini-app de carga |

### Clima
| Variable | Ejemplo |
| --- | --- |
| `WEATHERAPI_KEY` | clave de weatherapi.com |

### Variables (no secrets)
| Variable | Para qué |
| --- | --- |
| `DRY_RUN` | `true` durante la fase de sombra: los scripts corren pero no envían nada real. Se define en Settings → Secrets and variables → **Variables**, no en Secrets. El workflow de backup la ignora a propósito: su mail va solo al admin. |
