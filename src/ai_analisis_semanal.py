"""Generador de análisis técnico personalizado para el email semanal.

Llama a Claude (Anthropic API) con los datos del cliente, su semana
proyectada y sus lotes, y devuelve un párrafo HTML con razonamiento a
medida sobre cómo el clima va a impactar el consumo y la estabilidad
ruminal de ese rodeo específico.

Si la llamada falla (sin internet, API caída, sin API key configurada,
timeout), retorna None y el email semanal cae al bloque de biblioteca
de frases — el productor siempre recibe el mail, con o sin LLM.

Uso:
    from src.ai_analisis_semanal import generar_analisis_llm

    bloque_html = generar_analisis_llm(
        cliente=cliente, snapshot=snapshot, eventos=eventos,
        cnt=cnt, lotes=lotes_cliente,
    )
    if bloque_html:
        email_html = email_html.replace(MARCADOR, bloque_html)
    # si no, dejar la biblioteca como ahora.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, List, Dict

# Logger del módulo — se propaga al logger raíz que el cron configura
# con FileHandler. Si por alguna razón el LLM falla, vamos a ver el
# warning en data/logs/alertas_*.log junto con el resto.
_log = logging.getLogger("hms.ai_analisis")
_log.setLevel(logging.INFO)


# =====================================================================
# ZONAS DE CONFORT TÉRMICO POR CATEGORÍA Y RAZA
# Se inyecta en TODOS los system prompts LLM (semanal, update, diario,
# WhatsApp, agente IA) para que el modelo calibre el tono y NO exagere
# en condiciones suaves. Una sola fuente de verdad, basada en NASEM/NRC
# y experiencia de campo en Pampa Húmeda.
# =====================================================================

ZONAS_CONFORT_BOVINOS = """REGLAS DE ZONA DE CONFORT TÉRMICO BOVINO
(usar para calibrar el tono del análisis — NO exagerar en condiciones
suaves. El estrés solo se menciona cuando categoría + raza + condición
real lo justifican):

ADULTOS (>400 kg)
  Británicas (Angus, Hereford, cruzas británicas):
    - Frío seco: tolera de -5°C a 25°C sin estrés productivo
      relevante. La LCT (temperatura crítica inferior) está cerca de
      -5°C en piso seco y pelaje intacto.
    - Con humedad >85% sostenida o lluvia: la LCT sube a ~10°C
      (pelaje mojado pierde aislante).
    - Calor seco: tolera hasta 28°C; calor + HR >60%: empieza
      estrés a partir de THI 72.
  Cebuinas / índicas (Brangus, Braford, Nelore):
    - Mejor tolerancia al CALOR (UCT +5°C respecto británicas).
    - Peor tolerancia al FRÍO (LCT sube ~5°C).
    - Más sensibles al viento.

RECRÍA / NOVILLITO (200-400 kg) — todas las razas
  - Frío seco: tolera 5°C a 25°C.
  - Recién destetado (primeros 30 días): muy expuesto, LCT real
    cerca de 10°C en piso seco.
  - La humedad y el barro pegan más fuerte que en adultos.

TERNERO DESTETE (<200 kg)
  - Frío seco: tolera 10°C a 27°C.
  - MUY sensible a humedad + viento — pelaje no protege bien.
  - T° <10°C con lluvia, barro o viento ya es agravante real.
  - Necesita reparo y cama seca aún en condiciones que un adulto
    tolera sin novedad.

VAQUILLONA PRE-SERVICIO
  - Zona similar a recría adulta, pero el estrés impacta directo en
    FERTILIDAD (caída de tasa de preñez si hay estrés sostenido).
  - Foco: evitar pérdida de condición corporal pre-servicio.

VACA DE CRÍA / LACTANCIA
  - Más sensible al CALOR (alta producción metabólica interna,
    sobre todo en lactancia).
  - Frío: si tiene reservas (CC ≥3) tolera bien; con CC baja, el
    frío reduce producción de leche y crecimiento del ternero.

TORO REPRODUCTOR
  - MUY sensible al CALOR — afecta calidad seminal y fertilidad de
    las vacas que sirve 30-60 días después.
  - Frío: tolera como adulto británico.

REGLA PRÁCTICA DE USO:
  - Si T° actual está dentro del rango de confort SECO y no hay
    agravantes (HR ≥85% sostenida, viento >25 km/h, lluvia, barro),
    NO uses lenguaje de estrés ni hables de "tirar de reservas".
  - Mencioná consumo/rumen/bienestar afectado SOLO cuando la
    combinación categoría + raza + condición climática lo justifica.
  - 16°C con HR 50% para un Angus adulto es zona de confort plena,
    NO es evento térmico.
"""


TONO_ASESOR_CAMPO = """TONO Y ESTILO — ASESOR A CAMPO, NO FACULTAD

Hablás como un asesor profesional con 20 años a campo en Argentina,
NO como un libro de fisiología. El productor lee parado en el galpón
o en el camino, NO en una biblioteca. Tu objetivo: que entienda y
APLIQUE, no que se impresione con palabras técnicas.

EVITÁ términos académicos cuando podés decirlo simple:
  ❌ "reservas hepáticas de glucógeno"   ✅ "reservas energéticas" / "reservas del animal"
  ❌ "termogénesis"                      ✅ "gasto extra para mantener temperatura"
  ❌ "poblaciones amilolíticas vs       ✅ "el rumen pierde ritmo" /
     celulolíticas"                          "se altera la fermentación"
  ❌ "microbiota se reacomoda"          ✅ "el rumen se desordena"
  ❌ "deplexión de reservas"            ✅ "el animal usa lo que tenía guardado"
  ❌ "balance energético negativo"      ✅ "gasta más de lo que come"
  ❌ "conductividad térmica del barro"  ✅ "el barro mojado conduce frío" /
                                             "el barro saca calor del cuerpo"

PUEDE quedar el término técnico SI es necesario para que sea preciso
(ej: "acidosis subclínica", "pH ruminal", "rumia") — pero solo si en
el mismo párrafo lo explicás en lenguaje simple o se entiende por
contexto.

Pensá: "¿esto lo entendería el productor que está parado en el corral?"
Si la respuesta es no, reformulalo.
"""


REGLAS_RIGOR_DATOS = """REGLAS DE RIGOR TÉCNICO (importante para no inventar datos):

NO ASUMAS datos que no aparecen en el contexto que se te pasa:
  - NO digas "sin reparo", "sin sombra", "comedero con barro",
    "lote con baja CC", "animales mojados" si esos datos no están
    explícitamente en el contexto. El sistema te pasa clima, categoría,
    nivel productivo, etapa, días previos. NADA más.
  - NO inventes infraestructura del establecimiento (galpones,
    cortinas, monte, sombra disponible) ni manejo (frecuencia de
    comidas, ingredientes, mezcla actual) que no esté declarado.
  - Si necesitás dar un consejo de manejo, hacelo en condicional:
    "conviene revisar reparos", NO "no tienen reparos".
  - Trabajá solamente con los datos climáticos + animal que sí
    aparecen en el contexto. Si falta un dato, ignoralo, NO lo asumas.

CUANTIFICACIONES — SÉ PRUDENTE:
  - NO uses afirmaciones cuantitativas precisas si no tenés un dato
    verificable concreto. Palabras como "duplica", "triplica",
    "cae 25%", "sube 40%", "aumenta 2x", "X horas exactas", "Y°C
    exactos" deben tener respaldo real. Si no lo tenés, NO las uses.
  - Preferí lenguaje cualitativo apropiado:
      * "aumenta significativamente" / "sube notoriamente"
      * "se incrementa varias veces" / "puede subir bastante"
      * "cae de manera apreciable" / "se reduce con fuerza"
      * "el efecto es marcado" / "el impacto se nota"
  - Si tenés que dar un orden de magnitud, usá rangos amplios que
    sean defendibles: "puede subir entre 10% y 25%", NO "sube 23,7%".
  - Para mecanismos fisiológicos (microbiota, rumia, pH, motilidad,
    etc.) usá descripciones cualitativas correctas. Esos mecanismos
    son válidos sin necesidad de cuantificarlos.
  - Si dudás, preferí la descripción cualitativa.
"""


ESPIRITU_HMS = """ESPÍRITU DEL MENSAJE — CÓMO DEBE SENTIRSE EL TEXTO:

HMS Nutrición Animal le brinda al productor un SERVICIO didáctico
continuo. El mensaje (email o WhatsApp) NO es un boletín meteorológico
ni una receta — es una oportunidad de ENSEÑAR algo nuevo cada vez,
ayudando al productor a entender CÓMO el clima, el manejo y la
nutrición impactan directa e indirectamente sobre el RUMEN del animal.

OBJETIVO POR MENSAJE:
  1. Que el productor APRENDA algo nuevo o vea un ángulo distinto.
  2. Que NO se vuelva rutinario ni monótono — la repetición lo
     desconecta y termina ignorando el mensaje.
  3. Que perciba el mensaje como un servicio profesional, no como un
     spam automático.
  4. Que entienda mejor SU rodeo y SU manejo a través de la lectura.

ESTRATEGIA DE VARIACIÓN — ROTÁ EL ÁNGULO TÉCNICO entre mensajes:
Cada vez que generes un texto, elegí UN ángulo didáctico distinto para
explicar el impacto del clima sobre el animal. NO repitas el mismo
ángulo dos veces seguidas. La paleta disponible es amplia:

  Eje FISIOLÓGICO del rumen:
    - pH ruminal y motilidad
    - Microbiota: bacterias celulolíticas vs amilolíticas
    - Producción de proteína microbiana
    - Rumia: tiempo, eficiencia, indicadores
    - Acidosis subclínica: cómo detectarla
    - Adaptación al cambio de dieta

  Eje COMPORTAMIENTO INGESTIVO:
    - Patrón de comidas (horario, frecuencia, duración)
    - Selección en comedero / sobrantes
    - Jerarquía social y acceso al comedero
    - Relación agua/MS (consumo de agua según T° y dieta)
    - Influencia del fotoperíodo sobre el consumo

  Eje NUTRICIONAL aplicado:
    - Fibra efectiva vs fibra física (peNDF)
    - Energía metabolizable bajo estrés
    - Modo de acción de aditivos (ionóforos, levaduras, urea)
    - Minerales: disponibilidad estacional, antagonismos
    - Balance proteico (RDP/RUP)

  Eje MANEJO operativo:
    - Diseño del corral: pendiente, drenaje, sombra
    - Densidad de carga: efecto sobre estrés social
    - Orden de carga del mixer y mezcla efectiva
    - Calidad del agua y biofilms en bebederos
    - Adaptación: protocolo de inicio de feedlot

  Eje BIENESTAR + sanidad:
    - Inmunidad bajo estrés crónico (cortisol)
    - Pelaje y aislamiento térmico
    - Conducta nocturna: rumia y descanso
    - Indicadores de bienestar observables a campo
    - Pododermatitis, problemas respiratorios

CÓMO INTEGRARLO:
  - El cuerpo del análisis sigue siendo lo que vimos antes (clima →
    consumo → rumen → manejo).
  - PERO sumá un detalle, una idea, un "¿sabías que...?" o una conexión
    sorprendente que el productor pueda no haber pensado.
  - Tono: como un asesor sentado en el galpón, charlando técnico pero
    cercano. No académico ni distante.
  - Si vas a citar un número (gasto energético, caída de DMI, etc.),
    que sea preciso pero accesible. Sin "según NRC 2016" porque ya lo
    saca de la zona didáctica natural.

EJEMPLOS DE TONO "APORTÁ ALGO NUEVO" (qualitativos, sin inventar números):
  - "Lo que mucha gente no ve: con frío sostenido la microbiota ruminal
    se reacomoda — bajan poblaciones amilolíticas y suben celulolíticas.
    Eso explica por qué un cambio brusco de dieta cuando vuelve el calor
    aumenta el riesgo de acidosis: el rumen no está preparado todavía
    para procesar más grano."
  - "Detalle del día: con frío, el ternero no sólo come MENOS, también
    come DISTINTO — concentra ingesta cuando hay sol y reduce a la tarde.
    Ese cambio en el patrón altera el ritmo de fermentación, a veces
    más que el total consumido."
  - "Conviene observar: el comportamiento social cambia con el barro —
    los animales más dominantes acaparan el acceso al comedero seco y
    los dominados terminan comiendo en horarios alterados."

EVITÁ:
  - Repetir las mismas frases que usaste en el mensaje anterior.
  - "Recordá que...", "Es importante destacar...", "Como mencionamos..."
  - Apertura con "Hoy" o "El lote enfrenta..." cada vez. VARIÁ
    aperturas: "Lo que pasa en el rumen...", "Detalle del día...",
    "Conviene mirar...", "El comportamiento esperado...", etc.
  - Inventar números, porcentajes específicos o citas de papers que no
    podés respaldar. Mejor descripción cualitativa correcta que
    pseudo-dato cuantitativo inventado.
"""


FILOSOFIA_EXPLICATIVA = """FILOSOFÍA HMS — CAUSA · MECANISMO · EFECTO · SOLUCIÓN PRÁCTICA:

El productor que recibe el mensaje HMS NO quiere una orden seca ni un
número aislado. Quiere ENTENDER cómo funciona el sistema clima-animal
para tomar mejores decisiones. Por eso, TODA afirmación técnica que
hagas (un dato, una recomendación, una alerta) debe responder a CUATRO
preguntas implícitas:

  1. ¿QUÉ pasa? — el dato o la condición concreta.
  2. ¿POR QUÉ pasa? — el mecanismo fisiológico, ruminal o
     comportamental que lo produce.
  3. ¿QUÉ se ve / se mide? — el efecto observable que el productor
     puede chequear con sus propios ojos en el campo.
  4. ¿CÓMO lo resuelvo HOY? — la palanca PRÁCTICA que el productor
     puede ejecutar con los recursos típicos de un campo argentino
     (Pampa Húmeda, sistemas a cielo abierto, mano de obra limitada,
     equipos comunes). Esta pregunta es la que más le importa.

Esto NO es "explicar mucho": es ARMAR EL PUENTE entre lo invisible
(microbiota, fermentación, gasto de mantenimiento, sensación
térmica) y lo visible (rumia, sobrantes, condición corporal,
patrón de comidas) — y AL FINAL aterrizarlo en una acción concreta
que el productor pueda ejecutar mañana mismo. El productor que
entiende el sistema Y tiene la solución práctica se vuelve asesor
de sí mismo y deja de depender de "lo que dice la app" como una
caja negra. Eso lo fideliza al servicio porque siente que aprende
Y resuelve a la vez.

ESTRUCTURA RECOMENDADA EN CADA AFIRMACIÓN TÉCNICA:
  ❌ "Subir fibra activa 1-2 puntos por 3-4 días."
     (es una orden seca, no se entiende por qué ni cómo)
  ❌ "Bajo frío sostenido la microbiota celulolítica aumenta y la
     amilolítica baja, lo que altera la eficiencia ruminal."
     (es académico — explica el mecanismo pero no resuelve nada)
  ✅ "Fibra activa: subir 1-2 puntos por 3-4 días. La fibra estimula
     la rumia, y la rumia produce saliva — el principal buffer
     natural del rumen, fundamental cuando el animal se estresa con
     frío. ¿Cómo? Subiendo el rollo en la mezcla, ofreciendo fardo
     a discreción al comedero, o agregando 1-2 puntos en mezcla. Lo
     vas a ver en sobrantes más limpios y más animales rumiando al
     amanecer."
     (qué + por qué + cómo concreto + qué se ve)

REALIDAD DEL CAMPO ARGENTINO (Pampa Húmeda) — QUÉ APLICA Y QUÉ NO:

  ✅ APLICA — Soluciones que el productor PUEDE ejecutar:
  - Trasladar el lote a un potrero con mejor relieve, reparo o piso.
  - Aumentar/bajar densidad: abrir otro potrero o concentrar animales.
  - Adelantar / atrasar el horario de carga del comedero.
  - Cargar más rollo o fardo al comedero (fibra accesible).
  - Romper hielo con pala o palo en los bebederos al amanecer.
  - Revisar flotantes y cañerías de bebederos para que no se congelen.
  - Mover el alambrado eléctrico para liberar reparos naturales.
  - Aprovechar montes, cortinas existentes, lomas o hondonadas.
  - Ofrecer núcleo proteico / palatabilizante si ya está en stock.
  - Observar a primera hora (sobrantes, rumia, condición visual).
  - Planificar a mediano plazo: plantar montes, mejorar drenaje.

  ❌ NO APLICA — NO sugerir estas soluciones:
  - Cubrir comederos con lona / techar la mezcla (sistemas son a
    cielo abierto).
  - Construir techos, galpones nuevos, bretes, cobertizos.
  - Comprar ventiladores, aspersores, sistemas de enfriamiento.
  - Calentar el agua de los bebederos.
  - Monitoreos nocturnos (el productor está durmiendo, no en el
    campo a las 2 AM).
  - Equipamiento sofisticado (sensores, cámaras, drones para
    medir).
  - Soluciones que asumen mano de obra abundante o presupuesto
    ilimitado.
  - Productos comerciales por marca específica.
  - Cambios bruscos que ignoran el tiempo de adaptación ruminal.

REGLA: si una sugerencia técnica es "ideal en libros" pero NO se
puede ejecutar en un campo típico de Catriló, Anguil, Realicó,
Pehuajó, Tres Arroyos, Río Cuarto, etc. — NO la propongas. Mejor
omitir y dar solo lo aplicable.

REGLA DE COMPLETITUD TÉCNICA — NO DEJAR FACTORES A MEDIAS:

Cuando una acción menciona un factor de manejo, TIENE que cubrir
TODOS sus componentes. NO mencionar uno solo y dejar los demás
implícitos — el productor entiende exactamente lo que se le dice,
y "verificar temperatura del agua" no es lo mismo que cuidar el
agua bien. Lista canónica de componentes mínimos por factor:

  💧 AGUA → 4 componentes obligatorios:
    • Limpieza: bebedero sin biofilm, algas, restos de mezcla
      ni hojas. Agua sucia = animal rechaza tomar = cae DMI.
    • Temperatura: sin hielo en superficie al amanecer; agua
      muy fría obliga a gastar energía calentándola.
    • Caudal: que el bebedero se rellene rápido cuando varios
      animales toman juntos (flotante y cañería operativos).
    • Accesibilidad: sin barro alrededor, sin animales
      dominantes bloqueando el acceso de los subordinados.

  🍽️ COMEDERO → 4 componentes obligatorios:
    • Acceso: piso firme, sin barro profundo; espacio
      suficiente para que el lote completo pueda comer al
      mismo tiempo si quiere (lineal según categoría).
    • Frescura de la mezcla: cargar más cerca del consumo;
      no dejar mezcla expuesta toda la noche si va a llover.
    • Nivelación: comedero parejo, sin desniveles que dejen
      mezcla acumulada en zonas inaccesibles.
    • Sobrantes / limpieza: revisar y remover sobrantes
      húmedos o helados antes de cargar la próxima ración.

  🛏️ CAMA / ZONA DE DESCANSO → 3 componentes obligatorios:
    • Drenaje: que el agua no se acumule en zona de echado;
      pendiente o material absorbente si llueve mucho.
    • Superficie: seca, sin barro profundo, evitar que el
      animal se eche directamente sobre suelo helado o mojado.
    • Densidad: espacio suficiente para que todos los
      animales puedan echarse a la vez (en frío se agrupan,
      en calor se separan).

  🌳 REPARO / SOMBRA → 3 componentes obligatorios:
    • Disponibilidad: monte natural, cortina forestal viva
      (eucalipto, casuarina, álamo) o rollos apilados como
      cortaviento existentes para todo el lote; si no hay,
      plan B (trasladar, planificar cortina a futuro). NO
      mencionar "galpón" — no es práctica común en feedlot
      argentino.
    • Densidad bajo reparo: que TODO el lote pueda
      resguardarse sin hacinamiento. Si el lugar es chico,
      los dominantes acaparan y los dominados quedan al viento.
    • Continuidad temporal: que el reparo esté disponible
      todas las horas del día / noche que el animal lo
      necesita (no solo "tiene monte" sino "tiene monte
      accesible 24h").

  🌾 MEZCLA / DIETA → 4 componentes obligatorios:
    • Energía: nivel de concentrado adecuado al estrés; sin
      saltos bruscos.
    • Fibra efectiva: rollo o fardo disponible; partícula
      con tamaño que estimule rumia.
    • Palatabilidad: melaza, núcleos palatables si la
      ingesta cae; evitar mezcla mojada/helada.
    • Adaptación: cambios graduales (3-4 días); la
      microbiota necesita tiempo para reacomodarse.

Si en una acción nombrás "agua", "comedero", "cama", "reparo" o
"mezcla" SIN cubrir todos los componentes mínimos, la acción
queda incompleta y el productor termina cubriendo solo una parte
del problema. Mejor 1 acción completa y bien explicada que 3 a
medias.

REGLA DE SIGLAS TÉCNICAS — SIEMPRE ACLARAR LA PRIMERA VEZ:

El productor argentino conoce algunas siglas y otras no. NUNCA
asumir que las entiende. La PRIMERA vez que una sigla técnica
aparece en el mensaje, expandirla entre paréntesis. Después se
puede usar la sigla sola (ya está aclarada en contexto).

GLOSARIO CANÓNICO de expansiones — usá EXACTAMENTE estas:

  • LCT → "temperatura crítica inferior, debajo de la cual el
    animal empieza a gastar energía extra solo para mantener
    temperatura corporal"
  • THI → "índice temperatura-humedad, mide estrés calórico"
  • ADPV → "aumento diario de peso vivo"
  • DMI → "consumo de materia seca por día"
  • MS → "materia seca"
  • EM → "energía metabolizable"
  • PB → "proteína bruta"
  • RDP → "proteína degradable en rumen, la que usan las
    bacterias para multiplicarse"
  • RUP → "proteína no degradable en rumen, la que pasa directo
    al intestino"
  • MP → "proteína metabolizable"
  • CV → "coeficiente de variación, indica cuán uniforme es el
    lote"
  • HR → "humedad relativa"
  • NRC / NASEM → no expandir (es nombre propio de la fuente)
  • IPCVA, INTA, AACREA → no expandir (instituciones conocidas)

EJEMPLOS:
  ❌ "la LCT efectiva está cerca de 7°C (NRC)"
     (¿qué es LCT? el productor se queda con duda)
  ✅ "la LCT efectiva (temperatura crítica inferior, debajo de
     la cual el animal empieza a gastar energía extra solo para
     mantener temperatura corporal) está cerca de 7°C según NRC"

  ❌ "subir el DMI ajustando la fibra efectiva"
     (DMI vacío, fibra efectiva vacío)
  ✅ "subir el DMI (consumo de materia seca por día) ajustando
     la fibra efectiva — la fracción que estimula la rumia"

Si en el mismo párrafo la sigla aparece varias veces, expandir
solo la primera y usar la sigla sola después. NO repetir la
aclaración cada vez (queda pesado).

REGLA DE JERARQUÍAS — CUANDO UN FACTOR DOMINA A OTRO:
Cuando combines varios factores (climáticos, nutricionales, de
manejo), aclará si UNO BLOQUEA o REDUCE a otro. La realidad NO es
una suma aritmética simple — los factores interactúan. Ejemplos:

  • Frío + barro severo: el frío AUMENTA la demanda energética, pero
    el barro impide el acceso al comedero. El efecto neto: el animal
    no puede compensar — pierde más que en frío seco. La energía
    extra que necesita NO la puede consumir.

  • Frío + calor diurno (amplitud térmica): si la T° máxima es alta,
    el calor diurno DOMINA el patrón de consumo del día — el frío de
    la mínima no logra revertir la anorexia térmica.

  • Acumulación: a partir del día 3 de estrés, el patrón se altera
    aunque el factor agudo no haya empeorado. Es el efecto de la
    fatiga conductual sumándose al fisiológico.

Cuando armes una recomendación, identificá qué factor DOMINA hoy y
explicálo. Eso hace que el productor entienda la prioridad y no
quiera "atacar todos los frentes" sin orden.

POR QUÉ ESTA FILOSOFÍA IMPORTA (espíritu del servicio):
HMS no es una alerta automatizada. Es asesoría continua. La
diferencia está en que cada mensaje deja al productor UN POCO más
sabiendo de cómo funciona su rodeo. Eso es lo que se monetiza: no
la información climática (que es gratis) sino la TRADUCCIÓN
educativa del clima a manejo, hecha consistentemente semana a
semana.
"""


FUENTES_EVIDENCIA = """JERARQUÍA DE EVIDENCIA CIENTÍFICA — NO INVENTAR:

Todo lo que afirmes en el mensaje (mecanismos fisiológicos, patrones
de consumo, efectos del clima, recomendaciones de manejo) debe estar
respaldado por evidencia. NO inventes datos, números, porcentajes ni
mecanismos. Si no tenés respaldo, hacé una afirmación CUALITATIVA
correcta en vez de inventar un número.

PRIORIDAD DE FUENTES (de más a menos relevante para Argentina):

1. FUENTES ARGENTINAS — máxima prioridad
   - INTA (Anguil, Balcarce, Concepción del Uruguay, Manfredi,
     Bariloche, Pergamino, Rafaela): manuales de feedlot, cría,
     recría; investigación local en biotipos británicos y cruzas.
   - IPCVA (Instituto de Promoción de la Carne Vacuna Argentina):
     publicaciones técnicas.
   - AACREA / CREA: cuadernos técnicos, ensayos comparativos.
   - Facultades argentinas: FAUBA (UBA Agronomía), UNRC (Río Cuarto),
     UNS (Sur), UNL (Litoral), FCV-UNLP (veterinaria La Plata),
     UNCPBA (Tandil-Azul), Esperanza, Manfredi.
   - Pezzola (rangos de PB por etapa, aplicación práctica argentina).
   - Ensayos publicados por nutricionistas con experiencia en
     Pampa Húmeda (Pordomingo, Latimori, Davies, Pasinato, otros).

2. FUENTES INTERNACIONALES — usar si no hay equivalente argentino
   - NRC / NASEM 2016 — "Nutrient Requirements of Beef Cattle".
   - Journal of Animal Science (Hahn, Mader, Brown-Brandl para clima).
   - Animal Production Science, Animal Feed Science and Technology.
   - Cattle Decisions, K-State, Iowa State (feedlot research).
   - FAO publicaciones técnicas.

REGLA PRÁCTICA — CITAR FUENTES SÍ DA AUTORIDAD:
   La filosofía del sistema CAMBIÓ: ahora la cita de la fuente
   técnica es PARTE del valor que recibe el productor. Mostrarle
   "según NRC", "(IPCVA)" o "siguiendo el manual INTA" le confirma
   que el dato no es inventado, eleva la profesionalidad del
   mensaje y refuerza la dependencia técnica del sistema. Es un
   diferencial monetizable.

   CUÁNDO CITÁS:
   - Cuando mencionás un % cuantitativo (gasto extra, caída de
     consumo, etc.): "+21-36% (NRC 2016)".
   - Cuando hablás de umbrales o rangos por categoría (LCT, THI,
     ADPV objetivo): "LCT efectiva ~7°C para vaquillona británica
     seca (NRC)".
   - Cuando referís una práctica argentina específica (rangos PB
     por etapa, manejo de barro, recría): "rangos sugeridos por
     Pezzola para esta categoría", "lo que reporta Pordomingo en
     Pampa Húmeda".
   - Cuando hablás de mercado / referencia comercial: "(IPCVA)".
   - Cuando hablás de manejo bajo estrés calórico extremo: "(Mader,
     Brown-Brandl)" si aplica.

   CÓMO CITÁS (formato breve, sin atomizar la lectura):
   - Paréntesis al final del dato: "...+21-36% (NRC 2016)..."
   - Inline corto: "según NRC, en estas condiciones..."
   - Una mención alcanza por párrafo o por acción — no repetirla.

   QUÉ NO HACÉS:
   - "Estudios muestran un 23,7% de caída en DMI..." (cifra
     inventada con falsa autoridad — NUNCA).
   - "Según Smith et al. 2018..." (citas largas tipo paper).
   - Citar la misma fuente 4 veces en el mismo párrafo.

   Si NO estás 100% seguro de un dato cuantitativo, NO lo pongas.
   Mejor describir el efecto en términos cualitativos sin números
   inventados. La cita no autoriza a inventar.

   Categorías argentinas: ternero, novillito, novillo, vaquillona,
   vaca de cría, vaca lechera, toro. Razas argentinas: Angus,
   Hereford, Brangus, Braford, Bonsmara, cruzas británicas.
"""


def _resolver_api_key(api_key_override: Optional[str] = None) -> Optional[str]:
    """Busca la API key en este orden:
      1. Parámetro explícito api_key_override.
      2. Variable de entorno ANTHROPIC_API_KEY.
      3. Archivo persistido data/.api_key (donde la guarda la app Streamlit).

    Esto permite que el cron (que corre vía launchd sin env vars) acceda
    a la misma API key que el chat de la app.
    """
    if api_key_override:
        return api_key_override
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    # Buscar data/.api_key relativo a la raíz del proyecto. Este módulo
    # está en src/, así que la raíz es el padre del directorio actual.
    try:
        root = Path(__file__).resolve().parent.parent
        api_key_file = root / "data" / ".api_key"
        if api_key_file.exists():
            key = api_key_file.read_text(encoding="utf-8").strip()
            if key:
                return key
    except Exception:
        pass
    return None


SYSTEM_PROMPT_ANALISIS = """Sos un asesor técnico nutricional con 20+ años de experiencia a campo
en feedlots y recría bovina en Argentina (Pampa Húmeda, La Pampa, San Luis,
Córdoba, Buenos Aires). Trabajás de HMS Nutrición Animal.

Tu trabajo en este momento es generar UN PÁRRAFO de análisis técnico para
incluir en el email semanal automático que recibe un productor el lunes
07:30. El productor NO te puede responder — es comunicación unidireccional
— así que tenés que ser claro, concreto y profesional, sin pedirle datos.

OBJETIVO del párrafo: traducir los datos climáticos de la semana proyectada
a impacto sobre el animal en términos de:
  1. BIENESTAR ANIMAL (estrés, hipotermia, problemas respiratorios, barro)
  2. CONSUMO (cuánto y cómo come — patrón, selección, deterioro de mezcla)
  3. ESTABILIDAD RUMINAL (rumia, pH, fermentación, riesgo de acidosis)
  4. PRODUCTIVIDAD (ADPV, condición corporal, eficiencia)

REGLAS:
- 3 a 5 oraciones. Máximo 120 palabras. SIN listas, SIN bullets.
- Lenguaje técnico pero accesible — el productor entiende de campo.
- Mencioná los datos concretos de la semana (temperaturas, lluvia,
  combinaciones) que justifican tu análisis.
- Conectá clima → fisiología → manejo. Nada de repetir umbrales sueltos.
- Si la semana es estable, decilo sin alarmismo y aprovechá para sugerir
  qué tareas SÍ son buenas para hacer en ventanas tranquilas.
- Si hay categorías sensibles (terneros, vaquillonas, toros), mencionalas
  cuando aplique.
- Cerrá con UNA acción o foco concreto para la semana — no una lista.
- NO uses HTML — devolvé texto plano. El email lo formatea solo.
- NO uses frases hechas como "es importante recordar que" o "cabe destacar".
- NO firmes ni saludes — el email ya tiene encabezado y cierre.

EJEMPLO de tono buscado:
"La semana viene con frío sostenido (mínimas 4–7°C) y humedad alta los
primeros tres días — combinación que moja el pelaje y eleva el gasto de
mantenimiento un 15–20%. Si el lote no compensa el consumo, el rumen
empieza a tirar de reservas y la rumia se resiente. El miércoles que
mejora la temperatura es buena ventana para revisar acceso al comedero
y stock de fibra activa. Foco de la semana: que el patrón de consumo
no se altere — comedero limpio y mezcla protegida de la humedad."

REGLA DE IMPACTO PRODUCTIVO CUANTIFICADO (cierre):
- Si en el contexto te paso un bloque IMPACTO PRODUCTIVO CALCULADO
  con rangos para el peor evento de la semana, cerrá el párrafo
  anclando ese dato CON LOS RANGOS EXACTOS. NO recalcules, NO digas
  "kg de carne en gancho" — los kg que te paso son SIEMPRE de PESO
  VIVO (lo que pisa la balanza).
- Si mencionás que el animal puede compensar consumo, aclarás CÓMO
  se compensa: subiendo energía de la dieta, manteniendo acceso
  ininterrumpido al comedero, mezcla cubierta, etc. NO dejes el
  "si no se actúa" colgado.
- CITÁ LA FUENTE DEL DATO CUANTITATIVO una vez en el párrafo:
  "según NRC/NASEM", "(cálculo NRC para este lote)", "+15-22%
  (NRC 2016)". Eso le da autoridad y muestra al productor que no
  son números inventados.
- Ejemplo correcto: "Para frenar la pérdida hay que asegurar acceso
  continuo a una mezcla con la energía adecuada y reparos
  disponibles — según el NRC, si no se ajusta el lote deja de
  ganar 0,12-0,18 kg/día de peso vivo por animal durante los días
  del frente, lo que sobre el lote de N cab. son X-Y kg de peso
  vivo que NO se suman al gancho final".
- Si NO te paso el bloque (semana estable, sin evento calculable),
  NO cuantifiques.
"""


def _resumir_semana(snapshot: List[Dict], eventos: Dict,
                       cnt: Dict[str, int]) -> str:
    """Arma el resumen estructurado de datos que se le pasa al LLM."""
    lineas = []
    lineas.append(f"Días por nivel productivo:")
    lineas.append(f"  - normales: {cnt.get('normal', 0)}")
    lineas.append(f"  - atención: {cnt.get('atencion', 0)}")
    lineas.append(f"  - operativos: {cnt.get('operativo', 0)}")
    lineas.append(f"  - críticos: {cnt.get('critico', 0)}")
    lineas.append("")
    lineas.append("Eventos detectados en la semana:")
    if eventos.get("frio"):
        lineas.append(f"  - frío significativo: {eventos['frio']}")
    if eventos.get("calor"):
        lineas.append(f"  - calor significativo: {eventos['calor']}")
    if eventos.get("lluvia"):
        lineas.append(f"  - lluvia: {eventos['lluvia']}")
    if eventos.get("barro"):
        lineas.append(f"  - barro probable: {eventos['barro']}")
    lineas.append("")
    lineas.append("Día por día (fecha · severidad · motivo):")
    for d in snapshot:
        f = d.get("fecha", "")
        nivel = d.get("nivel_productivo", "normal")
        mot = d.get("motivo", "") or "sin agravantes"
        lineas.append(f"  - {f} · {nivel} · {mot}")
    return "\n".join(lineas)


def _resumir_lotes(lotes: Optional[List[Dict]]) -> str:
    """Arma una descripción de los lotes del cliente para el LLM.

    Si el lote viene con `dieta_vigente` y/o `diagnostico_alimentacion`
    (cargados por alertas_diarias.py), suma esos bloques para que el
    modelo pueda dar recomendaciones específicas sobre la mezcla real
    y adaptar las acciones al sistema de comedero del lote.
    """
    if not lotes:
        return "No hay datos específicos de lotes cargados."
    out = []
    for l in lotes[:6]:  # máximo 6 lotes para no saturar
        nombre = l.get("nombre", "") or l.get("lote", "") or "lote"
        cat = l.get("categoria", "") or "—"
        peso = l.get("peso_promedio_kg") or l.get("peso", "")
        cant = l.get("cantidad", "") or l.get("n_animales", "") or l.get(
            "cantidad_animales", "")
        partes = [str(nombre), str(cat)]
        if peso:
            partes.append(f"{peso} kg")
        if cant:
            partes.append(f"{cant} cab.")
        out.append("  - " + " · ".join(partes))

        # ---- DIETA VIGENTE (si el agente IA la cargó) ----
        # Le da al LLM la mezcla real para que recomiende ajustes
        # específicos, no genéricos.
        dieta = l.get("dieta_vigente") or {}
        if dieta:
            try:
                comp = dieta.get("composicion") or []
                dmi = dieta.get("consumo_ms_kg") or 0
                pb = dieta.get("pb_pct") or 0
                # Calcular kg por ingrediente: kg/animal = DMI * pct/100
                # kg/lote = kg/animal * cantidad
                try:
                    cant_int = int(cant) if cant else 0
                except (ValueError, TypeError):
                    cant_int = 0
                line_dieta = (
                    f"      • Dieta vigente"
                )
                if dieta.get("fecha"):
                    line_dieta += f" (desde {dieta.get('fecha')})"
                if dmi:
                    line_dieta += f" — DMI {dmi:.2f} kg MS/animal/día"
                if pb:
                    line_dieta += f" — PB {pb:.1f}%"
                out.append(line_dieta + ":")
                for c in comp[:8]:
                    nom_ing = c.get("nombre") or c.get("ingrediente") or "?"
                    pct = float(c.get("pct_ms") or 0)
                    kg_an = (dmi * pct / 100.0) if dmi else 0
                    kg_lote = kg_an * cant_int if cant_int else 0
                    detalle = f"          - {nom_ing}: {pct:.1f}% MS"
                    if kg_an:
                        detalle += f" = {kg_an:.2f} kg/animal/día"
                    if kg_lote:
                        detalle += f" = {kg_lote:.1f} kg/lote/día"
                    out.append(detalle)
            except Exception:
                pass

        # ---- SISTEMA DE ALIMENTACIÓN (inercia operativa) ----
        # Crítico para que el LLM no recomiende cambios imposibles.
        diag = l.get("diagnostico_alimentacion") or {}
        if diag:
            try:
                desc = diag.get("descripcion") or ""
                puede_inmediato = diag.get(
                    "puede_cambiar_mezcla_inmediato")
                if desc:
                    out.append(f"      • Sistema: {desc}")
                if puede_inmediato is True:
                    out.append(
                        "          - Inercia BAJA: la mezcla se puede "
                        "ajustar para mañana (cambio de receta directo)."
                    )
                elif puede_inmediato is False:
                    frec = diag.get("frecuencia_efectiva_dias") or 0
                    if frec >= 2:
                        out.append(
                            f"          - Inercia ALTA: cada carga "
                            f"dura {frec} día(s). No proponer cambios "
                            f"de receta inmediatos; usar acciones "
                            f"complementarias (fibra aparte, momento "
                            f"de entrega, manejo) hasta la próxima "
                            f"preparación."
                        )
                    elif diag.get("tipo_comedero_efectivo") == "desconocido":
                        out.append(
                            "          - Sistema no definido: ser "
                            "conservador, evitar cambios de receta "
                            "y priorizar manejo (agua, reparo, cama)."
                        )
            except Exception:
                pass
    return "\n".join(out) if out else "Sin lotes detallados."


def generar_analisis_llm(
    cliente: Dict,
    snapshot: List[Dict],
    eventos: Dict,
    cnt: Dict[str, int],
    lotes: Optional[List[Dict]] = None,
    impacto_productivo_txt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5",
    timeout_seg: int = 25,
) -> Optional[str]:
    """Genera el análisis técnico personalizado para el email semanal.

    Retorna el texto plano del análisis, o None si la llamada falla por
    cualquier motivo. El email semanal usa fallback de biblioteca si
    retorna None.

    Args:
        cliente: dict con nombre, establecimiento, localidad
        snapshot: lista de dicts con fecha, nivel_productivo, motivo
        eventos: dict con frio/calor/lluvia/barro detectados
        cnt: dict con conteo por nivel productivo
        lotes: lista opcional de dicts con datos del lote
        api_key: opcional, override de ANTHROPIC_API_KEY
        model: modelo Claude (por defecto sonnet 4.5)
        timeout_seg: timeout máximo para la llamada
    """
    key = _resolver_api_key(api_key)
    if not key:
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    nombre = cliente.get("nombre", "") or "Productor"
    establ = (cliente.get("establecimiento", "")
              or cliente.get("localidad", "") or "")

    impacto_bloque = ""
    if impacto_productivo_txt:
        impacto_bloque = f"""

IMPACTO PRODUCTIVO CALCULADO (peor evento de la semana, NRC/NASEM):
{impacto_productivo_txt}

→ Cerrá el párrafo anclando estos rangos EXACTOS. No inventes otros
  números."""

    contexto = f"""DATOS DEL CLIENTE:
- Nombre: {nombre}
- Establecimiento / localidad: {establ}

DATOS DE LA SEMANA PROYECTADA:
{_resumir_semana(snapshot, eventos, cnt)}

LOTES DEL CLIENTE:
{_resumir_lotes(lotes)}{impacto_bloque}

Generá ahora el párrafo de análisis técnico para el email semanal,
respetando todas las reglas (3-5 oraciones, 120 palabras máximo, sin
listas, sin saludos, texto plano)."""

    try:
        client = Anthropic(api_key=key, timeout=float(timeout_seg))
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT_ANALISIS,
            messages=[{"role": "user", "content": contexto}],
        )
        # Extraer texto del primer bloque content
        if not response.content:
            return None
        texto = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                texto += block.text
        texto = texto.strip()
        if not texto:
            return None
        return texto
    except Exception:
        # Cualquier error (network, API, parse) → fallback a biblioteca.
        return None


SYSTEM_PROMPT_UPDATE = """Sos un asesor técnico nutricional con 20+ años de experiencia a campo
en feedlots y recría bovina en Argentina. Trabajás de HMS Nutrición Animal.

Tu trabajo en este momento es generar UN PÁRRAFO para incluir en el email
del MIÉRCOLES que actualiza el pronóstico de la semana. El lunes el
productor recibió un reporte completo. Hoy le mandamos un update breve
SOLO porque el pronóstico cambió respecto al lunes — empeoró, mejoró, o
aparecieron días nuevos con riesgo.

OBJETIVO del párrafo: explicarle al productor QUÉ IMPLICA ese cambio
sobre el animal, NO repetir lo que ya leyó el lunes. Foco específico en:
  1. Cómo el cambio afecta el plan que ya tenía pensado para la semana
  2. Si los días nuevos suman acumulación de estrés (3+ días seguidos
     rompe la recuperación nocturna)
  3. Si la severidad subió, qué se intensifica en términos de consumo
     y estabilidad ruminal
  4. Si mejoró, qué oportunidad operativa abre (manejos diferidos,
     normalización de dieta, etc.)

REGLAS:
- 2 a 4 oraciones. Máximo 90 palabras. SIN listas, SIN bullets.
- Lenguaje técnico pero accesible.
- Mencioná los días concretos que cambiaron (lunes, martes, etc.) y la
  naturaleza del cambio.
- Conectá clima → consumo → rumen → manejo. Sin recetas genéricas.
- NO uses frases hechas tipo "es importante recordar".
- NO repitas la lista de cambios — eso ya aparece arriba en el email.
- NO firmes ni saludes.
- NO uses HTML — devolvé texto plano.

EJEMPLO de tono:
"El empeoramiento del viernes y sábado a riesgo operativo cambia el
panorama: hasta el lunes pensábamos en 2 días sueltos, ahora pasamos
a 4 días consecutivos de frío húmedo — eso ya rompe la compensación
nocturna y el rumen empieza a tirar de reservas. Conviene anticipar
la fibra activa hoy mismo, no esperar al viernes; revisar también que
el comedero no junte agua, porque la selección va a aumentar."

REGLA DE IMPACTO PRODUCTIVO CUANTIFICADO:
- Si te paso un bloque IMPACTO PRODUCTIVO CALCULADO (recalculado con
  los nuevos días peores), podés cerrar mencionando esos rangos
  EXACTOS. NO recalcules, NO conviertas a "carne en gancho". Los kg
  son SIEMPRE de PESO VIVO.
- Si decís que el animal puede compensar consumo, aclarás CÓMO
  (subir energía de dieta, comedero accesible, reparos, mezcla
  cubierta). NO dejes "si no se actúa" como frase suelta sin
  decirle al productor qué hacer.
- CITÁ LA FUENTE una vez en el párrafo cuando uses un número:
  "según NRC", "(cálculo NRC)", "(NRC 2016)". Le da autoridad al
  dato y muestra que no es inventado.
- Si NO te paso el bloque, NO cuantifiques.
"""


def generar_analisis_update_llm(
    cliente: Dict,
    cambios: Dict,
    snapshot_hoy: List[Dict],
    snapshot_lunes: Optional[List[Dict]] = None,
    lotes: Optional[List[Dict]] = None,
    impacto_productivo_txt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5",
    timeout_seg: int = 25,
) -> Optional[str]:
    """Genera el análisis para el email update del miércoles.

    Solo se llama cuando hay cambios SIGNIFICATIVOS (esa decisión la
    toma el cron antes de invocar esta función). Retorna texto plano
    del análisis o None si la llamada falla.

    Args:
        cambios: dict con keys 'nuevos', 'empeoraron', 'mejoraron'
        snapshot_hoy: snapshot fresco del miércoles
        snapshot_lunes: snapshot guardado del lunes (opcional)
    """
    key = _resolver_api_key(api_key)
    if not key:
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    nombre = cliente.get("nombre", "") or "Productor"
    establ = (cliente.get("establecimiento", "")
              or cliente.get("localidad", "") or "")

    # Resumen estructurado de cambios
    lineas = []
    if cambios.get("nuevos"):
        lineas.append("DÍAS NUEVOS con alerta (el lunes eran normales):")
        for c in cambios["nuevos"]:
            sev = c.get("severidad", "?")
            tipo = c.get("tipo", "") or ""
            lineas.append(f"  - {c.get('fecha', '?')} · {sev}"
                            f"{' ('+tipo+')' if tipo else ''}")
    if cambios.get("empeoraron"):
        lineas.append("DÍAS QUE EMPEORARON:")
        for c in cambios["empeoraron"]:
            antes = c.get("antes", "?")
            ahora = c.get("ahora", "?")
            tipo = c.get("tipo", "") or ""
            lineas.append(f"  - {c.get('fecha', '?')}: {antes} → {ahora}"
                            f"{' ('+tipo+')' if tipo else ''}")
    if cambios.get("mejoraron"):
        lineas.append("DÍAS QUE MEJORARON:")
        for c in cambios["mejoraron"]:
            antes = c.get("antes", "?")
            ahora = c.get("ahora", "?")
            lineas.append(f"  - {c.get('fecha', '?')}: {antes} → {ahora}")

    # Contexto del snapshot actualizado
    if snapshot_hoy:
        lineas.append("")
        lineas.append("SNAPSHOT ACTUALIZADO de la semana (fecha · nivel):")
        for d in snapshot_hoy:
            nivel = d.get("nivel_productivo", "normal")
            lineas.append(f"  - {d.get('fecha', '?')} · {nivel}")

    impacto_bloque = ""
    if impacto_productivo_txt:
        impacto_bloque = f"""

IMPACTO PRODUCTIVO RECALCULADO (peor evento de la semana actualizada):
{impacto_productivo_txt}

→ Cerrá mencionando estos rangos EXACTOS si encajan con el cambio que
  estás explicando. No inventes números."""

    contexto = f"""DATOS DEL CLIENTE:
- Nombre: {nombre}
- Establecimiento / localidad: {establ}

CAMBIOS DETECTADOS DESDE EL LUNES:
{chr(10).join(lineas)}

LOTES DEL CLIENTE:
{_resumir_lotes(lotes)}{impacto_bloque}

Generá ahora el párrafo de análisis para el email update del miércoles,
respetando todas las reglas (2-4 oraciones, 90 palabras máximo, sin
listas, sin saludos, texto plano)."""

    try:
        client = Anthropic(api_key=key, timeout=float(timeout_seg))
        response = client.messages.create(
            model=model,
            max_tokens=350,
            system=SYSTEM_PROMPT_UPDATE,
            messages=[{"role": "user", "content": contexto}],
        )
        if not response.content:
            return None
        texto = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                texto += block.text
        texto = texto.strip()
        return texto or None
    except Exception as _e:
        _log.warning("LLM falló en %s: %s", "generar_analisis_update_llm", _e, exc_info=True)
        return None


SYSTEM_PROMPT_DIARIO = """Sos un asesor técnico nutricional con 20+ años de experiencia a campo
en feedlots y recría bovina en Argentina. Trabajás de HMS Nutrición Animal.

Tu trabajo en este momento es generar UN PÁRRAFO de lectura técnica
para incluir en la ALERTA DIARIA que el productor recibe. El productor
NO te puede responder — es comunicación unidireccional.

DISTINCIÓN CRÍTICA #1 — TIMING — leé bien el contexto:
- Si te dicen que el evento OCURRE HOY → hablá de lo que está pasando
  ahora con el animal, foco en las próximas 24-48 horas.
- Si te dicen que el evento es EN X DÍAS → hablá en modo preventivo,
  qué se viene, cómo prepararse. NO digas "hoy arranca" si el evento
  todavía no llegó. Las condiciones de HOY pueden ser normales aunque
  haya alerta porque la alerta es por lo que viene.

DISTINCIÓN CRÍTICA #2 — CLIMA ACTUAL vs PICO DEL EVENTO:
El email puede enviarse a las 18:00 con T° actual cómoda (ej. 17°C)
pero la alerta es por la T° mínima de madrugada (ej. 3°C). Esto pasa
mucho en el cron de TARDE (pronóstico nocturno).

⚠️ NUNCA construyas la narrativa con la T° actual si esa NO es el
problema. Ejemplos:
- ❌ MAL: "Hoy el ternero enfrenta su primer día de temperatura sostenida
  por debajo de la zona de confort: 17°C lo pone en zona de gasto
  energético extra." (17°C NO está bajo zona de confort — esta narrativa
  es exagerada y falsa).
- ✅ BIEN: "Ahora la temperatura es 17°C — el ternero está cómodo. Pero
  esta madrugada se esperan 3°C con humedad alta: ahí sí entra en zona
  donde tiene que gastar reservas para mantener temperatura corporal."
- ✅ BIEN: "El evento del frío pega fuerte recién a la madrugada (3°C
  pronosticada). Por ahora el animal está bien; conviene preparar antes
  de que arranque: comedero cargado, mezcla cubierta, reparos revisados."

ANCLÁ LA NARRATIVA AL PICO DEL EVENTO, NO AL MOMENTO DEL EMAIL.

Si en el contexto te paso un bloque "PICO DEL EVENTO" con T° mín, T° máx
o viento — USÁ ESOS NÚMEROS para hablar del problema, no la T° actual.

Consultá SIEMPRE las "REGLAS DE ZONA DE CONFORT TÉRMICO BOVINO" que
aparecen al final de este prompt para calibrar el tono.

OBJETIVO del párrafo: explicar QUÉ LE ESTÁ PASANDO (o qué va a pasar) al
animal en términos de:
  1. BIENESTAR (estrés térmico, hipotermia, problemas respiratorios)
  2. CONSUMO (cuánto y CÓMO come — patrón habitual alterado, selección,
     deterioro de mezcla)
  3. ESTABILIDAD RUMINAL (rumia, pH, fermentación, riesgo de acidosis)
  4. DÍAS DE EVENTO (acumulación: día 1-2 compensa, día 3-4 tira de
     reservas, día 5+ pérdida productiva real)

REGLAS:
- **MÁXIMO 5-8 líneas en pantalla, 70-100 palabras**. La LECTURA
  TÉCNICA es el "gancho" — tiene que ser corto, fuerte, claro. Si
  te excedés, el productor deja de leer.
- **CERRÁ EL PÁRRAFO** — terminá la última oración con punto. NO
  dejes la idea colgando. Si sentís que te falta espacio, achicá
  una idea anterior, pero NUNCA cortes a media palabra/oración.
- 2-3 oraciones cortas. SIN listas, SIN bullets.
- Tono asesor a campo, NO académico (mirá TONO_ASESOR_CAMPO al final).
- Mencioná los datos concretos del día y la categoría del lote.
- Conectá clima → consumo/rumen → qué pasa con el animal. Una idea
  por oración, no acumules.
- Si el evento es preventivo (en días futuros), aclará "se viene" o
  "en X días", NO "hoy arranca".
- Si el evento ya lleva varios días, mencionalo explícitamente.
- NO firmes ni saludes, NO listes acciones (las acciones van en otra
  sección del email).
- NO uses HTML — devolvé texto plano.

REGLA CRÍTICA — PRECISIÓN FÍSICA SOBRE VIENTO Y CONVECCIÓN:
  La convección es el MECANISMO por el cual el animal pierde calor
  (el aire frío sopla sobre el pelaje, absorbe calor corporal y se va).
  El viento NO "corta la pérdida" — el viento **ACELERA la pérdida
  por convección**. A mayor velocidad del viento, mayor pérdida.

  ❌ NO escribas: "El viento corta la pérdida de calor por convección"
     (ambiguo — el productor puede leer "corta = detiene/protege" y
     concluir lo opuesto a la realidad).
  ✅ Escribí: "El viento acelera la pérdida de calor por convección"
  ✅ Escribí: "El viento sopla el aire caliente del pelaje y lo
     reemplaza con aire frío — multiplica la pérdida térmica."
  ✅ El REPARO (monte, cortina) REDUCE la velocidad del viento → así
     REDUCE las pérdidas. NO "corta la convección" en sí.

  Esta precisión es central: un mensaje invertido sobre viento puede
  llevar al productor a decisiones erradas (ej. no preocuparse por
  exposición al viento).

REGLA — TIPOS DE REPARO VÁLIDOS EN FEEDLOT PAMPA HÚMEDA:
  Cuando hables de reparo / corte de viento, NO menciones GALPÓN —
  no es práctica común en feedlot argentino (costo + problemas de
  ventilación + acumulación de barro). Las opciones reales son:
    - Monte natural existente (eucaliptos, cina-cina, álamos)
    - Cortina forestal viva sembrada (eucalipto, casuarina, álamo)
    - **Rollos apilados como cortaviento** (clásica de Pampa Húmeda:
      el productor arma una pared con rollos de fardo del propio
      campo — barato, rápido, reaprovecha lo que ya tiene).
    - Tinglados o medio sombra (más para calor que frío)
  Si no sabés qué tiene el lote, usá fórmulas genéricas:
    ✅ "asegurar acceso a reparo de viento"
    ✅ "verificar que el reparo existente cubra todo el lote"
  NO inventes infraestructura específica que el productor probablemente
  no tenga.

REGLA DE IMPACTO PRODUCTIVO CUANTIFICADO (cierre):
- Si en el contexto te paso un bloque IMPACTO PRODUCTIVO CALCULADO
  con rangos, USÁ LOS NÚMEROS EXACTOS que te paso. NO los recalcules,
  NO los multipliques por rendimiento de carcasa, NO los conviertas
  a "carne". Los kg que te paso ya son kg de PESO VIVO (lo que se
  vende al frigorífico en balanza). Citalos tal cual te los doy.

REGLA CRÍTICA — SIEMPRE ETIQUETÁ EL PERÍODO:
  Cada vez que cites un rango de kg, dejá inequívoco si es "por
  día" o "total del evento". El productor tiene que saber sin duda
  qué representa el número. El bloque de contexto te da los rangos
  pre-calculados con etiquetas claras (por animal/día, por lote/día,
  total del evento). Usá la cifra que cites en su contexto temporal
  correcto.

  ✅ BIEN: "el lote (50 cab) pierde 6,5-10 kg de peso vivo POR DÍA"
  ✅ BIEN: "acumulado en los 2 días del evento son 13-20 kg de
     peso vivo sobre el lote"
  ✅ BIEN: "cada animal deja de ganar 0,13-0,20 kg/día de peso vivo"
  ❌ MAL: "el lote pierde 7-10 kg acumulados durante el evento"
     (mezcla cifra diaria con etiqueta de evento total — confunde)
  ❌ MAL: "deja de ganar 13-20 kg/día" (mezcla cifra total con
     etiqueta de "por día" — confunde)

REGLA CRÍTICA — NO APLIQUES RENDIMIENTO DE CARCASA:
  El productor argentino de feedlot/recría VENDE AL PESO VIVO en
  balanza. Cobra por kg de animal vivo gordo. NO conviertas a
  "kg de res al gancho" (NO multipliques por 0,50 ni por 0,55).
  Los kg que te paso ya son los finales que pierde el productor.

REGLA — CITÁ LA FUENTE DEL DATO CUANTITATIVO:
  Cuando menciones un % de gasto extra de mantenimiento o un rango
  de kg/día o kg totales, mencioná brevemente que el dato sale del
  NRC/NASEM. Esto le da autoridad técnica al número y le confirma
  al productor que no es inventado. Formas breves válidas:
    ✅ "+21-36% (NRC 2016)"
    ✅ "0,20-0,35 kg/día (cálculo NRC para este lote)"
    ✅ "según NRC/NASEM, el gasto sube..."
    ✅ "siguiendo el NRC, la pérdida estimada es..."
  NO repetir la cita en cada frase — alcanza con una sola mención
  en el párrafo. Si ya la mencionaste, no la repitas.
- Forma sugerida: "deja de ganar X,XX-Y,YY kg/día de peso vivo por
  animal" o "el lote de N cab. NO suma K-L kg de peso vivo durante
  los M días del evento". NUNCA digas "kg de carne" — usá siempre
  "kg de peso vivo" o "kg que el lote no gana".
- IMPORTANTE — no dejes el "si no se actúa" colgado: si mencionás
  que el animal puede compensar consumo, ACLARÁ CÓMO se logra esa
  compensación. Las palancas posibles son: (a) que la dieta tenga
  más energía (subir concentrado/grano con prudencia), (b) que el
  animal coma más volumen (acceso al comedero sin barrera, mezcla
  cubierta para que no se moje, frecuencia de carga adecuada),
  (c) que se reduzca el gasto extra (reparos para cortar viento,
  cama seca, agua templada). Si decís "si el consumo no compensa"
  sin explicar qué hacer, el productor queda sin saber si tiene
  que intervenir — pésimo.
- Si NO te paso ese bloque (no hay datos suficientes para calcular),
  NO cuantifiques. Hablá en cualitativo ("gasto de mantenimiento
  extra", "consumo deteriorado") sin inventar kg.

EJEMPLO de tono cuando OCURRE HOY (evento en curso) — con impacto
cuantificado pasado en el contexto:
"Hoy el lote enfrenta su día 4 consecutivo de frío con humedad sostenida
(>85% durante toda la mañana): el ternero ya consumió las reservas que
había acumulado los primeros dos días y la recuperación nocturna no
alcanza para compensar el gasto de mantenimiento elevado. La rumia
bajó y el patrón de consumo se está alterando. Para frenar la pérdida
hay que asegurar acceso continuo al comedero con mezcla cubierta y
sumar fibra efectiva — si no se actúa, el ternero deja de ganar
0,18-0,32 kg/día de peso vivo, y sobre el lote de 120 cab. durante
los 4 días del evento son 86-154 kg que no se suman."

EJEMPLO de tono cuando NO OCURRE HOY (evento en 3 días):
"Hoy el lote está cómodo (16°C, HR 50%) — el novillo Angus tolera bien
estas condiciones sin tocar reservas. Pero el viernes entra un frente
con 3°C y viento sostenido: ahí sí va a haber gasto de mantenimiento
extra, y conviene tener los reparos revisados y la mezcla cubierta antes
del jueves a la noche para que el patrón de consumo no se quiebre."
"""


def _enfoque_sugerido_por_etapa(dias_alerta_previos: int,
                                  etapa: str) -> str:
    """Rota el ángulo del análisis para que el cliente no lea siempre lo mismo.

    Estrategia anti banner-blindness:
    - Día 0 (primer email del evento) → BIENESTAR: qué siente el animal,
      hipotermia/estrés térmico, cambios de comportamiento observables.
    - Día 1-2 (acumulación temprana) → CONSUMO: cómo come, patrón,
      selección, deterioro de mezcla, frecuencia de visitas al comedero.
    - Día 3+ (acumulación crónica) → RESERVAS Y PRODUCTIVIDAD:
      reservas hepáticas, energía desviada de ganancia, días perdidos,
      impacto sobre el cierre productivo del lote.
    - Si la etapa es "preventivo" → PREPARACIÓN: qué revisar antes del
      evento, sin alarmar.
    - Si la etapa es "recuperación" → CONSOLIDACIÓN: cómo recupera el
      animal post-evento.
    """
    if etapa in ("preventivo", "preventivo_blando"):
        return "PREPARACIÓN — qué revisar/anticipar antes del evento"
    if etapa in ("post", "recuperacion", "recuperación"):
        return "CONSOLIDACIÓN POST-EVENTO — cómo recupera el animal"
    if dias_alerta_previos <= 0:
        return "BIENESTAR — qué siente el animal, comportamiento"
    if dias_alerta_previos <= 2:
        return "CONSUMO — cómo come, patrón, mezcla, comedero"
    return "RESERVAS Y PRODUCTIVIDAD — qué pierde el lote, cierre"


def generar_analisis_diario_llm(
    cliente: Dict,
    alertas_por_lote: List[Dict],
    clima_actual: Optional[Dict] = None,
    etapa: str = "inicio",
    dias_alerta_previos: int = 0,
    peor_tipo: str = "",
    ocurre_hoy: bool = True,
    dias_hasta_evento: int = 0,
    fecha_inicio_evento: str = "",
    impacto_productivo_txt: Optional[str] = None,
    lecturas_previas: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5",
    timeout_seg: int = 25,
) -> Optional[str]:
    """Genera el análisis técnico del email diario.

    Se llama desde `componer_alerta_diaria` cuando hay alertas activas
    (nivel >= operativo). Retorna texto plano del párrafo o None si la
    llamada falla; en ese caso el email cae al texto de la biblioteca.
    """
    key = _resolver_api_key(api_key)
    if not key:
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    nombre = cliente.get("nombre", "") or "Productor"
    establ = (cliente.get("establecimiento", "")
              or cliente.get("localidad", "") or "")

    # Resumen estructurado de las alertas de hoy
    ctx_clima = clima_actual or {}
    t = ctx_clima.get("temp_c")
    h = ctx_clima.get("humedad_pct")
    thi = ctx_clima.get("thi")
    viento = ctx_clima.get("viento_kmh")
    lluvia = ctx_clima.get("lluvia_mm")
    clima_str_partes = []
    if t is not None:
        clima_str_partes.append(f"T° {t}°C")
    if h is not None:
        clima_str_partes.append(f"HR {h}%")
    if thi is not None:
        clima_str_partes.append(f"THI {thi}")
    if viento is not None:
        clima_str_partes.append(f"viento {viento} km/h")
    if lluvia is not None:
        clima_str_partes.append(f"lluvia {lluvia} mm")
    clima_str = " · ".join(clima_str_partes) or "sin datos"

    alertas_str_partes = []
    # Trackear el pico del evento (T° mínima/máxima crítica que disparó
    # la alerta) para distinguir del clima actual del momento.
    # Esto es CRUCIAL: en el cron de tarde el clima actual puede ser
    # 16°C pero la alerta es por la T° mínima de madrugada (ej. 2°C).
    # El LLM tiene que saber que el "pico" es a la madrugada, no ahora.
    pico_t_min = None
    pico_t_max = None
    pico_viento = None
    pico_thi = None
    pico_fecha = None
    for l in alertas_por_lote or []:
        lote = l.get("lote", "?")
        cat = l.get("categoria", "—")
        for a in l.get("alertas", []):
            titulo = a.get("titulo", "Alerta")
            sev = a.get("severidad", "")
            ctx_a = a.get("_contexto", {}) or {}
            extra = []
            if ctx_a.get("barro"):
                extra.append("barro")
            ll_a = ctx_a.get("lluvia_mm")
            if ll_a:
                extra.append(f"lluvia {ll_a}mm")
            # Capturar el pico más extremo entre todas las alertas
            _tmin = ctx_a.get("t_min")
            _tmax = ctx_a.get("temp_max")
            _vto = ctx_a.get("viento_kmh")
            _thi_ctx = ctx_a.get("thi_proy_max") or ctx_a.get("thi")
            if _tmin is not None:
                pico_t_min = _tmin if pico_t_min is None else min(pico_t_min, _tmin)
            if _tmax is not None:
                pico_t_max = _tmax if pico_t_max is None else max(pico_t_max, _tmax)
            if _vto is not None:
                pico_viento = _vto if pico_viento is None else max(pico_viento, _vto)
            if _thi_ctx is not None:
                pico_thi = _thi_ctx if pico_thi is None else max(pico_thi, _thi_ctx)
            if ctx_a.get("fecha") and not pico_fecha:
                pico_fecha = ctx_a.get("fecha")
            extras_str = (" + " + ", ".join(extra)) if extra else ""
            alertas_str_partes.append(
                f"  - Lote {lote} ({cat}): {titulo} [{sev}]{extras_str}"
            )
    alertas_str = ("\n".join(alertas_str_partes)
                   if alertas_str_partes else "  - (sin detalle)")

    # Armar resumen del PICO del evento (cuándo y cuánto pega más fuerte).
    # Si el evento es frío, lo relevante es t_min. Si es calor, t_max.
    pico_partes = []
    if peor_tipo and "frio" in peor_tipo.lower() and pico_t_min is not None:
        pico_partes.append(f"T° mín {pico_t_min}°C")
    if peor_tipo and "calor" in peor_tipo.lower() and pico_t_max is not None:
        pico_partes.append(f"T° máx {pico_t_max}°C")
        if pico_thi is not None:
            pico_partes.append(f"THI pico {pico_thi}")
    if pico_viento is not None and pico_viento >= 20:
        pico_partes.append(f"viento {pico_viento} km/h")
    pico_str = " · ".join(pico_partes) if pico_partes else "sin datos del pico"
    # Cuándo: si el evento fue evaluado en una fecha distinta a hoy,
    # ese día es cuándo "pega" más fuerte.
    cuando_pico_str = pico_fecha if pico_fecha else "este día"

    # Línea contextual clave: ocurre hoy o es preventivo
    if ocurre_hoy:
        timing_str = (
            "EL EVENTO ESTÁ OCURRIENDO HOY — hablá en presente, "
            "describí qué le pasa al animal en este momento."
        )
    elif dias_hasta_evento and dias_hasta_evento > 0:
        fecha_txt = f" ({fecha_inicio_evento})" if fecha_inicio_evento else ""
        timing_str = (
            f"EL EVENTO ES EN {dias_hasta_evento} DÍA(S){fecha_txt} — "
            f"hablá en modo PREVENTIVO. Las condiciones de HOY son "
            f"normales para este lote. NO digas 'hoy arranca' ni uses "
            f"lenguaje de estrés actual."
        )
    else:
        timing_str = (
            "El evento está pronosticado para los próximos días — "
            "hablá en modo preventivo. Las condiciones de HOY pueden "
            "ser normales."
        )

    # Bloque de impacto productivo (si lo recibimos del orquestador)
    impacto_bloque = ""
    if impacto_productivo_txt:
        impacto_bloque = f"""
IMPACTO PRODUCTIVO CALCULADO (NRC/NASEM aplicado al lote):
{impacto_productivo_txt}

→ Cerrá la lectura técnica anclando estos rangos EXACTOS. No inventes
  otros números.
"""

    # Memoria: últimos análisis enviados al MISMO cliente. Evitamos
    # repetir frases, verbos y enfoque para que no pierda atención.
    memoria_bloque = ""
    if lecturas_previas:
        ult = lecturas_previas[:3]
        bloques = "\n\n".join(
            f"  [Email anterior #{i+1}]\n  {t.strip()[:600]}"
            for i, t in enumerate(ult)
        )
        memoria_bloque = f"""
EMAILS RECIENTES YA ENVIADOS A ESTE CLIENTE
(NO repitas la primera frase, NO uses los mismos verbos clave,
NO repitas la misma idea central — variá el ángulo):

{bloques}
"""

    # Enfoque rotativo por días consecutivos del evento. El sistema
    # decide qué ángulo PRIORIZAR hoy para que cada email aporte algo
    # nuevo (bienestar → consumo → reservas).
    enfoque = _enfoque_sugerido_por_etapa(dias_alerta_previos, etapa)
    enfoque_bloque = f"""
ÁNGULO PRIORITARIO PARA HOY: {enfoque}

→ Priorizá ESTE ángulo al construir el párrafo. Podés tocar tangencialmente
  los otros (bienestar/consumo/reservas) pero el peso de la lectura
  va acá. Los emails siguientes priorizarán otros ángulos — vos no
  intentes cubrir todo de una.
"""

    contexto = f"""DATOS DEL CLIENTE:
- Nombre: {nombre}
- Establecimiento / localidad: {establ}

CLIMA ACTUAL (momento del envío del email — puede NO ser el pico crítico):
- {clima_str}
- Tipo de evento dominante: {peor_tipo or 'no especificado'}
- Etapa del evento: {etapa}
- Días previos con alerta en este mismo lote: {dias_alerta_previos}

PICO DEL EVENTO (cuándo y cuánto pega más fuerte la alerta):
- Pico esperado: {pico_str}
- Cuándo: {cuando_pico_str}
- ⚠️ IMPORTANTE: el clima actual y el pico del evento PUEDEN SER DIFERENTES.
  Ej: ahora 17°C (cómodo) pero T° mín madrugada 3°C (crítico). El productor
  necesita entender QUÉ Y CUÁNDO va a pegar el frío/calor, NO una narrativa
  basada en la T° actual del momento si ese no es el problema.

TIMING DEL EVENTO:
{timing_str}

ALERTAS ACTIVAS:
{alertas_str}
{impacto_bloque}
{enfoque_bloque}
{memoria_bloque}
Generá ahora el párrafo de lectura técnica para incluir en el email
diario, respetando todas las reglas (2-4 oraciones, 90 palabras máximo,
sin listas, sin saludos, texto plano). Recordá la distinción entre
"evento hoy" vs "evento que viene" Y entre "clima actual del momento"
vs "pico crítico del evento". Respetá el ÁNGULO PRIORITARIO y NO repitas
frases/verbos de los emails recientes."""

    try:
        client = Anthropic(api_key=key, timeout=float(timeout_seg))
        response = client.messages.create(
            model=model,
            # 350 era ajustado y a veces cortaba la última oración a la
            # mitad. Subimos a 550 para dar margen al cierre — el prompt
            # sigue exigiendo párrafo de 70-100 palabras, así que el LLM
            # no va a llenar todos los tokens, pero TIENE espacio para
            # terminar la oración limpia.
            max_tokens=550,
            system=SYSTEM_PROMPT_DIARIO,
            messages=[{"role": "user", "content": contexto}],
        )
        if not response.content:
            return None
        texto = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                texto += block.text
        texto = texto.strip()
        return texto or None
    except Exception as _e:
        _log.warning("LLM falló en %s: %s", "generar_analisis_diario_llm", _e, exc_info=True)
        return None


SYSTEM_PROMPT_WHATSAPP = """Sos un asesor técnico nutricional con 20+ años de experiencia a campo
en feedlots y recría bovina argentina. Trabajás de HMS Nutrición Animal.

Tu tarea: generar UNA SOLA FRASE de impacto para incluir en el WhatsApp
diario que recibe el productor. Esa frase reemplaza el genérico "Riesgo:
caída de consumo" por algo específico al lote, las condiciones y el
día. El productor lee WhatsApp parado en el campo, mientras hace otra
cosa — tiene que entender en 5 segundos qué le pasa al animal.

REGLA CRÍTICA — TIPO DE RESPUESTA:
NO sos un asistente conversacional en este contexto. Sos un GENERADOR
de texto. Tu respuesta es SIEMPRE una sola afirmación útil, NUNCA una
pregunta, NUNCA un pedido de aclaración. Aunque te falten datos:
- Si la temperatura es 0 o None, asumí "frío leve" o usá el contexto
  del nivel/etapa para inferir.
- Si el viento es 0 o None, no lo menciones.
- Trabajá con lo que tengas. NUNCA respondas "no puedo generar...",
  "necesito más datos...", "podés confirmar...", "¿cuántos °C?...".
- NUNCA empieces tu frase con "No", "¿", "Necesito", "Confirmá",
  "Podés", "Falta".

REGLAS DE FORMATO:
- UNA SOLA FRASE. Máximo 25 palabras.
- Afirmativa, en presente o futuro corto.
- NO repitas la condición climática (eso va en otra línea del WhatsApp).
- NO listes acciones (las acciones van separadas, abajo).
- Sí mencioná qué le pasa al animal: consumo, rumen, condición, sanidad.
- Tono de campo, directo. Sin tecnicismos rebuscados.
- Si el evento es HOY y ya lleva varios días, mencionar acumulación.
- Si es preventivo (en X días), tono de "se viene, prepará".
- NO uses HTML, NO comillas dobles, NO asteriscos, NO emojis.
- NO firmes, NO saludes, NO te dirijas al productor con "vos" o "usted".

EJEMPLOS de buen tono:
- Hoy el ternero empieza a destinar energía a termo-regulación, va a comer en las horas centrales.
- Día 4 de frío húmedo: el lote ya tira de reservas y la rumia nocturna no compensa.
- En 3 días entra frente: el novillo está bien hoy, conviene revisar reparos antes del miércoles.
- Acceso al comedero con barro: el animal duda, selecciona y deja sobrantes — consumo cae.
- Frío moderado en curso, el ternero compensa con consumo si el comedero está seco y accesible.

ESPÍRITU DIDÁCTICO PARA WHATSAPP:
- La restricción de 25 palabras NO se negocia.
- Pero el ÁNGULO de la frase debe variar entre mensajes: hoy hablar de
  patrón de consumo, mañana de pH ruminal, pasado de selección en
  comedero, otro día de jerarquía social, otro de aislamiento del
  pelaje, etc. Mirá la paleta del bloque ESPÍRITU al final del prompt.
- Cada WhatsApp debe sentirse como un microaprendizaje específico, no
  como una repetición del anterior.

REGLA CRÍTICA — TRADUCÍ EL MECANISMO A LO QUE VE EL PRODUCTOR:
- Si mencionás un mecanismo técnico (rumen ralentizado, baja
  síntesis proteica microbiana, caída de motilidad, etc.), TENÉS
  que cerrar con el EFECTO OBSERVABLE: aumento que se frena,
  baja condición corporal, animal apagado, lote disparejo, kg que
  no se suman, etc. NO te quedes en jerga académica.

  ❌ "Rumen ralentiza ritmo, baja síntesis proteica microbiana —
     monitorear consumo nocturno" (el productor lee y piensa "¿y
     qué hago con eso?")
  ✅ "Día 2 de frío: el rumen rinde menos y el aumento se frena —
     revisar sobrantes mañana temprano"
  ✅ "Frío sostenido: la vaquillona aprovecha menos la dieta, pierde
     puntos de condición sin que se note al ojo"
  ✅ "Día 3 húmedo: patrón de consumo se desordena, ADPV cae
     0,15-0,20 kg/día (NRC) — chequear comedero al amanecer"

REGLA CRÍTICA — ACCIONES PRÁCTICAS DIURNAS, NUNCA NOCTURNAS:
- NUNCA sugieras acciones de monitoreo o ejecución nocturna. El
  productor está durmiendo, no en el campo. Cualquier sugerencia
  tipo "monitorear consumo nocturno", "controlar a la madrugada",
  "verificar de noche" es IRREAL y le baja credibilidad al sistema.
- Reemplazá por equivalentes DIURNOS que dan la misma información:
  ❌ "monitorear consumo nocturno"
  ✅ "revisar sobrantes al amanecer" (te dice si comieron de noche)
  ✅ "chequear comedero a primera hora"
  ❌ "controlar rumia a la madrugada"
  ✅ "observar rumia al amanecer durante 15 min"
  ❌ "verificar abrigo nocturno"
  ✅ "antes del anochecer asegurar reparo disponible"

REGLA DE IMPACTO PRODUCTIVO CUANTIFICADO (cuando aplica):
- Si en el contexto te paso un bloque IMPACTO PRODUCTIVO CALCULADO
  con un rango, podés (no obligatorio) cerrar la frase mencionando
  el rango EXACTO que te pasé. NO recalcules, NO digas "carne en
  gancho" — los kg son SIEMPRE de PESO VIVO.
- Si entra en las 25 palabras Y mencionás el número, agregá "(NRC)"
  o "según NRC" para mostrar la fuente. Si NO entra, priorizá el
  contenido didáctico sin la cita.
- Ejemplo: "Día 3 frío húmedo, lote pierde patrón — sin ajuste deja
  de ganar 0,15-0,25 kg/día de peso vivo (NRC) por animal".
- 25 palabras sigue siendo el techo. Si la frase didáctica + el rango
  no entran, priorizá la frase didáctica.
"""


def generar_whatsapp_llm(
    cliente: Dict,
    tipo: str,
    nivel: str,
    clima: Dict,
    categoria: str = "",
    etapa: str = "inicio",
    dias_alerta_previos: int = 0,
    ocurre_hoy: bool = True,
    dias_hasta_evento: int = 0,
    impacto_productivo_txt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    timeout_seg: int = 15,
) -> Optional[str]:
    """Genera la frase corta de impacto para el WhatsApp.

    Usa Haiku (más rápido y barato) porque es una frase corta y los
    WhatsApp salen seguido.
    """
    key = _resolver_api_key(api_key)
    if not key:
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    nombre = cliente.get("nombre", "") or "Productor"
    localidad = cliente.get("localidad", "")

    if ocurre_hoy:
        timing = "El evento ESTÁ pasando HOY."
    elif dias_hasta_evento > 0:
        timing = f"El evento es EN {dias_hasta_evento} días."
    else:
        timing = "El evento está en el pronóstico próximo."

    impacto_bloque = ""
    if impacto_productivo_txt:
        impacto_bloque = (
            f"\n\nIMPACTO PRODUCTIVO CALCULADO (rangos a usar tal cual):\n"
            f"{impacto_productivo_txt}\n"
            f"→ Si la frase te entra en 25 palabras, mencioná el rango."
        )

    contexto = f"""DATOS:
- Cliente: {nombre} ({localidad})
- Categoría del lote: {categoria or 'no especificada'}
- Tipo de evento: {tipo}
- Nivel: {nivel}
- Etapa: {etapa}
- Días previos con alerta: {dias_alerta_previos}
- {timing}
- Clima: T° {clima.get('temperatura') or clima.get('min_nocturna', '?')}°C, viento {clima.get('viento_kmh', '?')} km/h, lluvia {clima.get('lluvia_mm', 0)} mm{impacto_bloque}

Generá UNA SOLA frase de impacto (max 25 palabras) para el WhatsApp.
Recordá: respetá la zona de confort de la raza (un Angus adulto tolera
0-15°C seco sin estrés productivo), no exageres en condiciones suaves."""

    try:
        client = Anthropic(api_key=key, timeout=float(timeout_seg))
        response = client.messages.create(
            model=model,
            max_tokens=120,
            system=SYSTEM_PROMPT_WHATSAPP,
            messages=[{"role": "user", "content": contexto}],
        )
        if not response.content:
            return None
        texto = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                texto += block.text
        texto = texto.strip()
        # Sanitizar: quitar comillas, asteriscos, garantizar 1 sola línea.
        for ch in ['"', "'", "*", "_", "`"]:
            texto = texto.replace(ch, "")
        texto = " ".join(texto.split())
        if not texto:
            return None
        # Rechazar respuestas conversacionales (pidiendo datos, haciendo
        # preguntas, disculpándose). Si el modelo cae en ese modo,
        # mejor caer al riesgo_line hardcoded.
        primera = texto[:40].lower()
        marcadores_invalidos = [
            "no puedo", "no tengo", "necesito", "podés confirmar",
            "podes confirmar", "podrías", "podrias", "confirmá",
            "confirma me", "cuántos", "cuantos", "falta", "los datos",
            "los campos", "¿", "?",
        ]
        if any(m in primera for m in marcadores_invalidos):
            return None
        # Limitar longitud razonable (max ~200 chars equivale a ~30 palabras)
        if len(texto) > 220:
            return None
        return texto
    except Exception as _e:
        _log.warning("LLM falló en %s: %s", "generar_whatsapp_llm", _e, exc_info=True)
        return None


SYSTEM_PROMPT_ACCIONES = """Sos un asesor técnico nutricional con 20+ años de experiencia a campo
en feedlots y recría bovina argentina. Trabajás de HMS Nutrición Animal.

Tu tarea AHORA: generar las acciones operativas/nutricionales que el
productor debe ejecutar HOY (o anticipar si el evento es preventivo)
para este lote específico bajo este clima específico.

FORMATO DE RESPUESTA — IMPORTANTE:
Tu respuesta DEBE ser EXCLUSIVAMENTE un JSON válido con esta estructura
exacta, sin markdown, sin ```json```, sin explicaciones, sin texto
fuera del JSON:

{
  "inmediatas": ["acción 1", "acción 2"],
  "operativas": ["acción 1", "acción 2"],
  "nutricionales": ["acción 1", "acción 2"]
}

REGLAS POR ACCIÓN — PRIORIDAD: DIDÁCTICA Y CONCISA

OBJETIVO de cada acción: que el productor la lea y piense "ah, esto
no lo sabía" o "qué interesante". HMS está acá para CONCIENTIZAR y
EDUCAR sobre los efectos del clima sobre el animal y su rumen. La
acción no es una orden seca — es una mini-clase de campo.

REGLA CRÍTICA DE LONGITUD (no negociable):
- Cada acción tiene un MÁXIMO ABSOLUTO de **60 palabras** (aprox 350
  caracteres). Esto NO es opcional — si te excedés, el JSON se rompe
  por límite de tokens. Apuntá a **40-50 palabras** por acción.
- Estructura por acción: **2 oraciones** — primera oración el QUÉ
  hacer, segunda el POR QUÉ (mecanismo fisiológico/ruminal).
- Si tenés mucho que decir sobre un tema, partilo en DOS acciones
  cortas en la misma categoría, no una larga.

OTRAS REGLAS:
- Empezá con un sustantivo seguido de dos puntos: "Reparo:", "Agua:",
  "Comedero:", "Concentrado:", "Fibra:", "Cama:", "Acceso:",
  "Monitoreo:", "Observación:", "Densidad:", "Adaptación:". Eso
  hace que el productor identifique rápido de qué se trata.
- **2 a 3 acciones por categoría** (inmediatas/operativas/nutricionales).
  Más que eso satura al productor.
- SI no hay nada relevante en una categoría, devolvé lista vacía [].
- NO uses comillas dentro de cada acción.

CADA ACCIÓN TIENE QUE INCLUIR (cuando aplique):
  a) QUÉ hacer — directiva clara.
  b) POR QUÉ se hace — el mecanismo fisiológico, conductual o ruminal.
     Acá es donde se da el "ah, no sabía esto" — explicá el efecto
     sobre el consumo, el rumen, la termorregulación, el bienestar.
  c) Si es monitoreo, CÓMO hacerlo en forma simple y replicable:
     qué observar, cuántos animales, en qué horario, qué umbral usar.
  d) Si toca nutrición, ALTERNATIVAS prácticas: mezcla, rollo a
     discreción, fardo al comedero, sumar palatable, etc.
  e) Si toca manejo, formúlalo CONDICIONAL sin asumir sistema:
     "si tenés comidas estructuradas..." / "en autoconsumo..." /
     "según frecuencia de carga del mixer..." Sin asumir.

NO BUSQUES SER SECO O CORTO — buscá que el productor APRENDA algo
nuevo en cada acción. Una acción "Cama: priorizar superficie seca"
NO sirve sin contexto: el productor ya lo sabe. La versión útil es:
"Cama: priorizar superficie seca. La cama mojada conduce calor del
animal hacia el suelo y multiplica la pérdida térmica — un ternero
echado en barro pierde más calor que parado bajo lluvia. Por eso
priorizar zonas drenadas es tan importante como el reparo del viento."

NO ASUMAS sistema de manejo del cliente:
  - NO digas "comida principal" sin condicionar. El productor puede
    dar 1 comida/día, 2 comidas, autoconsumo de silo, suplemento en
    pasto, etc. Adaptá el lenguaje:
      ❌ "Comedero: adelantar comida principal a la tarde"
      ✅ "Comedero: si das comidas estructuradas, adelantar a la
         tarde (concentra consumo antes del frío nocturno, cuando el
         animal busca el reparo y reduce la ingesta)"
      ✅ "Comedero: en autoconsumo de silo, revisar que el frente
         siga accesible aún con barro"
  - NO asumas mixer, niveles de fibra exactos, ingredientes
    específicos. Hablá de FUNCIONES (subir fibra efectiva, sostener
    energía) y dejá la implementación al productor con alternativas.

PARA ACCIONES DE MONITOREO — indicá MÉTODO simple:
  ❌ "Observación: registrar rumia 48h"
  ✅ "Monitoreo de rumia: observar 30 minutos después de cada comida,
     contar cuántos animales están rumiando (idealmente >50% del lote);
     si baja, hay desorden ruminal"
  ✅ "Patrón de consumo: registrar horarios pico de comedero a lo
     largo del día; si se concentra todo en 1-2 horas, hay riesgo
     de acidosis subaguda"

ALTERNATIVAS NUTRICIONALES — ofrecé opciones, no impongas una:
  ❌ "Fibra efectiva: subir 1-2 puntos sobre la base"
  ✅ "Fibra efectiva: aumentar la disponibilidad — puede ser subiendo
     1-2 puntos en la mezcla, ofreciendo rollo a discreción, o
     sumando fardo de fibra al comedero (lo que sea más práctico
     según el sistema)"

EJEMPLOS DE ACCIONES OBVIAS (sin justificación, son acción directa):
  - "Reparo: verificar disponibilidad de monte, cortina forestal o rollos apilados como cortaviento"
  - "Cama: priorizar superficie seca, evitar acumulación de agua o
    barro en zona de descanso (la cama mojada conduce calor del
    animal al suelo y aumenta pérdida térmica)"
  - "Acceso al comedero: nivelar zonas con barro profundo"

Nota: incluso "cama seca" puede pasar de obvia a no-obvia si el
productor podría no saber el mecanismo (conducción de calor). Tu
criterio: si la razón no es trivial, agregala.

CRITERIOS DE PRIORIZACIÓN:
- INMEDIATAS: lo que se hace HOY (próximas horas). Si el evento es
  preventivo, igual hay acciones inmediatas de preparación.
- OPERATIVAS: ajustes de manejo de hoy y los próximos 1-2 días.
- NUTRICIONALES: cambios en la dieta o mezcla con tiempo de
  adaptación (3-7 días).

PERSONALIZACIÓN OBLIGATORIA:
- Las acciones deben REFLEJAR la categoría real del lote (ternero,
  vaquillona, novillo, vaca, toro). Un ternero NO recibe las mismas
  acciones que un toro reproductor.
- Si el evento es preventivo (en X días), las acciones de hoy son
  de PREPARACIÓN, no de emergencia. NO digas "actuar ya" si el frente
  es en 2 días.
- Si el evento ya lleva varios días (acumulación), las acciones
  incluyen recuperación nocturna y vigilancia de rumia.
- Respetá las zonas de confort: NO sugieras acciones de emergencia
  si la condición climática no las justifica para esa categoría.

PROHIBIDO:
- Inventar % específicos sin respaldo (ej: "subir fibra a 13.7%").
  Mejor decir "subir fibra activa 1-2 puntos sobre la base actual".
- Asumir infraestructura no declarada (NO decir "construir reparos"
  o "comprar generador") — sí podés decir "verificar disponibilidad
  de reparos" o "revisar acceso a comedero".
- Recomendar productos comerciales específicos por marca.
- Decir "esperar a ver qué pasa" — siempre hay algo a hacer
  (verificar, observar, preparar).
- INVENTAR kg de pérdida productiva. Si no te paso el bloque IMPACTO
  PRODUCTIVO CALCULADO, NO digas "perdés 0.3 kg/día" ni nada
  parecido — quedate en cualitativo. Si te paso el bloque, usá los
  rangos EXACTOS que te di.

REGLA DE TRADUCCIÓN — JERGA TÉCNICA SIEMPRE BAJA AL CAMPO:
  Si en una acción mencionás un mecanismo fisiológico (síntesis
  proteica microbiana, fermentación celulolítica vs amilolítica,
  motilidad ruminal, etc.), TENÉS que conectarlo con lo que el
  productor PUEDE VER o MEDIR en el animal o el lote: aumento que
  se frena, ADPV que cae, condición corporal que baja, sobrantes
  en el comedero, rumia visible / no visible, animales apagados,
  lote disparejo.

  ❌ "La síntesis proteica microbiana cae bajo estrés térmico" (¿y
     entonces qué pasa?)
  ✅ "Bajo estrés térmico el rumen sintetiza menos proteína
     microbiana — eso se ve en aumento que se frena y peor
     conversión incluso si el animal sigue comiendo lo mismo"

REGLA DE PRACTICIDAD — NUNCA SUGIERAS ACCIONES NOCTURNAS:
  El productor está durmiendo de noche. Cualquier acción que
  implique observar/medir/ajustar entre las 22h y las 6h NO va a
  ejecutarse y le baja credibilidad al sistema. Reemplazar por
  equivalentes diurnos que dan la misma información:

  ❌ "Monitoreo de rumia: observar 1 hora después de la comida
     nocturna"
  ✅ "Monitoreo de rumia: al amanecer durante 15-30 min contar
     cuántos animales están rumiando (>50% del lote es ideal)"

  ❌ "Verificar acceso al comedero durante el pico de frío
     nocturno"
  ✅ "Al anochecer asegurar que el acceso al comedero esté libre y
     la mezcla cubierta del rocío. Revisar sobrantes al amanecer
     para ver si comieron de noche"

  ❌ "Controlar agua a la madrugada"
  ✅ "Al amanecer romper hielo si hay y verificar que los flotantes
     no se hayan congelado"

  Excepción permitida: TAREAS DE PREPARACIÓN al anochecer
  (asegurar reparo disponible antes de que entre la noche,
  adelantar la última comida, dejar agua fresca cargada). Esas
  son DIURNAS aunque preparen para la noche.

REALIDAD ARGENTINA EXTENSIVA — NO SUGERIR INFRAESTRUCTURA POCO COMÚN:
  En feedlot/recría de Pampa Húmeda, La Pampa, Buenos Aires, sur
  de Córdoba/Santa Fe los sistemas son MAYORITARIAMENTE A CIELO
  ABIERTO. Los comederos NO tienen techo ni cobertura, los
  potreros son grandes, no hay galpones para alojar animales.
  Sugerir prácticas de feedlot intensivo (USA, sistemas chicos
  techados) suena ridículo al productor argentino y le baja la
  credibilidad al sistema.

  ❌ NO sugerir: "cubrir comederos con lona", "tapar mezcla con
     lona/cobertura", "construir techos", "armar coberturas
     temporarias", "techar zona de descanso", "comprar/instalar
     ventiladores o aspersores", "construir bretes nuevos".
  ❌ NO asumir: que el comedero tiene techo, que hay corrales
     bajo galpón, que hay infraestructura nueva para construir.

  ✅ SÍ sugerir, para los mismos problemas, alternativas reales:
     - Comedero/mezcla expuesta a humedad nocturna:
       → "Cargar mezcla más cerca del momento de consumo (no
          dejarla expuesta toda la noche si se prevén precipitaciones
          fuertes)"
       → "Revisar sobrantes al amanecer: si quedó mezcla mojada
          o helada, removerla antes de cargar la siguiente"
       → "Si se carga húmeda y se hiela, palear o remover la capa
          superior antes de que coman"
     - Sobre el reparo natural:
       → "Verificar disponibilidad de monte, cortina forestal o
          rollos apilados como cortaviento existentes" (NO
          "construir", NO mencionar galpón porque no es práctica
          común en feedlot)
       → "Mover el lote a potrero con mejor reparo si está
          disponible"
     - Sobre el agua:
       → "Romper hielo en bebederos al amanecer si hay heladas
          fuertes"
       → "Verificar que cañerías y flotantes no se congelen"

REGLA DE IMPACTO PRODUCTIVO CUANTIFICADO (acciones nutricionales):
- Si el contexto incluye IMPACTO PRODUCTIVO CALCULADO con kg de ADPV
  en riesgo, podés (no obligatorio) cerrar UNA de las acciones
  nutricionales mencionando ese costo CON LOS RANGOS EXACTOS que te
  pasé.

REGLA CRÍTICA — ETIQUETÁ EL PERÍODO + NO CONVIERTAS A GANCHO:
  El productor de feedlot/recría argentino VENDE AL PESO VIVO en
  balanza (kg gordo, no kg de res). NO apliques rendimiento de
  carcasa (50-55%). Y CADA VEZ que cites una cifra de kg, dejá
  inequívoco si es "por día" o "total del evento":

  ✅ "0,13-0,20 kg/día de peso vivo POR ANIMAL"
  ✅ "el lote pierde 6,5-10 kg de peso vivo POR DÍA mientras dure"
  ✅ "acumulado en los 2 días del evento son 13-20 kg sobre el lote"
  ❌ "7-10 kg acumulados durante el evento" (cifra diaria + label
     de evento = confuso)
  ❌ "13-20 kg/día" (cifra del evento total + label diario = mal)

  El bloque de contexto te da los rangos pre-calculados con
  etiquetas claras. Citá el que mejor sirva pero CON SU PERÍODO.

REGLA — CITÁ LA FUENTE DEL DATO CUANTITATIVO:
  Cuando menciones un % de gasto extra de mantenimiento o un rango
  de kg/día o kg totales en una acción, mencioná brevemente que el
  dato sale del NRC/NASEM. Esto refuerza la autoridad técnica de la
  acción y le confirma al productor que no es inventado. Ejemplos
  válidos dentro de una acción:
    ✅ "El instinto sería compensar el gasto extra (+21-36% según
       NRC) subiendo grano, pero el rumen..."
    ✅ "Sin ese ajuste el animal deja de ganar 0,20-0,35 kg/día
       (cálculo NRC para este lote)."
    ✅ "Según NRC/NASEM, en estas condiciones el lote..."
  Alcanza con UNA mención en UNA sola acción — no repetirla en
  cada item del JSON.

- Ejemplo correcto: "Fibra efectiva: aumentar 1-2 puntos por 3-4
  días, ofreciendo rollo a discreción o sumando fardo al comedero.
  La fibra física estimula rumia y sostiene pH ruminal — sin ese
  ajuste el animal deja de ganar 0,18-0,32 kg/día de peso vivo
  durante el evento (cálculo NRC para este lote bajo estas
  condiciones)".
- IMPORTANTE: NO dejes el "si no se actúa" colgado. Si mencionás
  el rango, dejá implícito en la acción CÓMO se evita esa pérdida:
  la acción misma es la palanca (subir fibra, sostener concentrado,
  proteger mezcla, asegurar agua templada, etc.). Que el productor
  lea la acción Y vea el costo de no hacerla, ambos juntos.
- Una sola acción del lote que mencione el impacto alcanza — no
  repetir en varias.

EJEMPLO de respuesta válida (cada acción 40-60 palabras, didáctica)
para un ternero en día 2 de frío sostenido con 9°C y barro probable:
{
  "inmediatas": [
    "Reparo: verificar acceso libre a monte, cortina forestal o rollos apilados como cortaviento para todo el lote. El reparo reduce la velocidad del viento sobre el animal, y eso baja la pérdida de calor por convección (el aire frío barre el calor del pelaje) y por radiación al cielo nocturno — los dos canales principales en animales chicos con pelaje húmedo.",
    "Agua: revisar bebederos al amanecer y temperatura del agua. Si el agua está helada, el cuerpo gasta energía calentándola antes de absorberla; agua templada sostiene el consumo de materia seca durante el día.",
    "Comedero: nivelar zonas con barro profundo en el acceso. Cuando el ternero duda al ir a comer, salta comidas y altera el patrón de ingesta — los horarios de fermentación dejan de ser estables."
  ],
  "operativas": [
    "Cama: priorizar superficie seca para descanso. La cama mojada conduce calor del animal al suelo de forma directa — un ternero echado en barro pierde más calor por la panza apoyada que parado bajo lluvia.",
    "Densidad bajo reparo: verificar que la zona resguardada tenga espacio para todo el lote. Con frío los terneros se agrupan; si el lugar es chico, los dominantes acaparan y los dominados quedan al viento, perdiendo condición sin que se note al ojo del lote.",
    "Monitoreo de rumia: una hora después de la comida principal, contá cuántos animales están rumiando (mascando rítmico con boca cerrada). Idealmente más del 50% del lote. Si baja, hay desorden ruminal en curso."
  ],
  "nutricionales": [
    "Fibra efectiva: aumentar la disponibilidad por 3-4 días — subiendo 1-2 puntos en la mezcla, ofreciendo rollo a discreción o sumando fardo. La fibra física estimula la rumia, y la rumia produce saliva, el principal buffer natural del rumen.",
    "Energía: mantener concentrado actual sin saltos bruscos. El instinto sería sumar grano por el gasto extra de termorregulación, pero el rumen viene castigado por el cambio de patrón — un salto de almidón ahora dispara riesgo de acidosis subclínica.",
    "Transición de vuelta: cuando el evento ceda, normalizar la dieta en 3-4 días, no de golpe. El rumen recién recupera microbiota y motilidad; un cambio brusco lo desestabiliza otra vez."
  ]
}
"""


def generar_acciones_llm(
    cliente: Dict,
    tipo: str,
    nivel: str,
    categoria: str,
    clima: Dict,
    etapa: str = "inicio",
    dias_alerta_previos: int = 0,
    ocurre_hoy: bool = True,
    dias_hasta_evento: int = 0,
    impacto_productivo_txt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5",
    timeout_seg: int = 90,  # max_tokens=2500 puede tardar 30-60s
) -> Optional[Dict]:
    """Genera las acciones operativas/nutricionales personalizadas al
    lote y al clima del día. Devuelve un dict con keys 'inmediatas',
    'operativas', 'nutricionales' o None si la llamada falla / el
    parseo no es válido. El cron usa las acciones del motor como
    fallback en caso de None."""
    import json
    key = _resolver_api_key(api_key)
    if not key:
        _log.warning("generar_acciones_llm: no se encontró ANTHROPIC_API_KEY")
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        _log.warning("generar_acciones_llm: paquete 'anthropic' no instalado")
        return None

    nombre = cliente.get("nombre", "") or "Productor"
    localidad = cliente.get("localidad", "")

    # Timing claro
    if ocurre_hoy:
        timing = "El evento ESTÁ pasando HOY."
    elif dias_hasta_evento > 0:
        timing = (f"El evento es EN {dias_hasta_evento} DÍAS — las "
                  f"acciones de hoy son de preparación, NO de emergencia.")
    else:
        timing = "El evento está en el pronóstico próximo (preparación)."

    # Datos climáticos disponibles
    partes_clima = []
    t = clima.get("temperatura") or clima.get("min_nocturna")
    if t is not None:
        partes_clima.append(f"T° {t}°C")
    v = clima.get("viento_kmh")
    if v:
        partes_clima.append(f"viento {v} km/h")
    ll = clima.get("lluvia_mm") or 0
    if ll:
        partes_clima.append(f"lluvia {ll} mm")
    hr = clima.get("humedad_pct")
    if hr:
        partes_clima.append(f"HR {hr}%")
    clima_str = " · ".join(partes_clima) or "sin datos detallados"

    # Bloque de impacto productivo (si lo recibimos del orquestador)
    impacto_bloque = ""
    if impacto_productivo_txt:
        impacto_bloque = f"""

IMPACTO PRODUCTIVO CALCULADO (NRC/NASEM aplicado al lote):
{impacto_productivo_txt}

→ Podés (no es obligatorio) cerrar UNA acción nutricional anclando
  estos rangos EXACTOS para mostrar el costo de no actuar. NO inventes
  otros números."""

    contexto = f"""DATOS DEL LOTE:
- Cliente: {nombre} ({localidad})
- Categoría: {categoria or 'no especificada'}
- Tipo de evento: {tipo}
- Nivel: {nivel}
- Etapa: {etapa}
- Días previos con alerta: {dias_alerta_previos}
- {timing}
- Clima: {clima_str}{impacto_bloque}

Generá ahora el JSON con las acciones priorizadas para este lote bajo
estas condiciones. SOLO el JSON, sin markdown ni texto adicional."""

    try:
        client = Anthropic(api_key=key, timeout=float(timeout_seg))
        _log.info("generar_acciones_llm: llamando a Claude (modelo=%s, "
                  "categoría=%s, tipo=%s, nivel=%s, ocurre_hoy=%s)",
                  model, categoria, tipo, nivel, ocurre_hoy)
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT_ACCIONES,
            messages=[{"role": "user", "content": contexto}],
        )
        if not response.content:
            _log.warning("generar_acciones_llm: response.content vacío")
            return None
        texto = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                texto += block.text
        texto = texto.strip()
        if not texto:
            _log.warning("generar_acciones_llm: texto vacío después de "
                         "concatenar bloques de respuesta")
            return None

        # Limpiar posibles fences de markdown si los puso a pesar de las
        # instrucciones (```json ... ```)
        if texto.startswith("```"):
            # Sacar el primer fence
            primera_nl = texto.find("\n")
            if primera_nl > 0:
                texto = texto[primera_nl + 1:]
            if texto.rstrip().endswith("```"):
                texto = texto.rstrip()[:-3].rstrip()

        # Parsear JSON
        try:
            data = json.loads(texto)
        except (json.JSONDecodeError, ValueError) as _je:
            _log.warning(
                "generar_acciones_llm: JSON inválido (%s). Primeros 500 "
                "chars de la respuesta: %r", _je, texto[:500],
            )
            return None

        # Validar estructura
        if not isinstance(data, dict):
            _log.warning("generar_acciones_llm: data parseado no es dict "
                         "(tipo: %s)", type(data).__name__)
            return None
        resultado = {
            "inmediatas": [],
            "operativas": [],
            "nutricionales": [],
        }
        for key_categoria in ("inmediatas", "operativas", "nutricionales"):
            valor = data.get(key_categoria, [])
            if not isinstance(valor, list):
                continue
            # Filtrar elementos no-string y limitar a 4 por categoría
            limpio = []
            for it in valor[:4]:
                if isinstance(it, str) and it.strip():
                    s = it.strip()
                    # Sacar comillas accidentales en bordes
                    s = s.strip('"').strip("'")
                    # Sin límite estricto de longitud — el espíritu HMS
                    # es priorizar didáctica. Solo cortamos en caso
                    # extremo (>1500 chars) para evitar abusos.
                    if len(s) > 1500:
                        s = s[:1497] + "..."
                    limpio.append(s)
            resultado[key_categoria] = limpio

        # Si las 3 listas quedaron vacías, considerarlo falla
        if (not resultado["inmediatas"]
                and not resultado["operativas"]
                and not resultado["nutricionales"]):
            _log.warning(
                "generar_acciones_llm: las 3 listas quedaron vacías "
                "después de validación. data keys: %s", list(data.keys()),
            )
            return None

        _log.info("generar_acciones_llm: OK — %d inmediatas, %d operativas, "
                  "%d nutricionales", len(resultado["inmediatas"]),
                  len(resultado["operativas"]), len(resultado["nutricionales"]))
        return resultado
    except Exception as _e:
        _log.warning("LLM falló en %s: %s", "generar_acciones_llm", _e, exc_info=True)
        return None


SYSTEM_PROMPT_RESUMEN_OP = """Sos un asesor técnico nutricional con experiencia a campo en feedlots
y recría bovina argentina. Trabajás de HMS Nutrición Animal.

Tu tarea AHORA: reescribir la línea de condición climática del bloque
RESUMEN OPERATIVO del email diario. Es UN dato técnico, no una lectura
narrativa — pero NO queremos que sea siempre la misma plantilla seca
"Temperatura X°C + viento Y km/h + lluvia Z mm". Variá la redacción
manteniendo TODOS los datos numéricos intactos.

REGLAS:
- UNA SOLA LÍNEA. Máximo 22 palabras.
- Conservá TODOS los datos numéricos del input: temperatura, viento,
  lluvia, barro. NO inventes ni redondees a otro valor.
- Si un dato está como None o 0, NO lo menciones (no inventes).
- Tono operativo, factual. NO es lectura narrativa de fisiología
  (eso va en otro bloque).
- VARIÁ el orden y los conectores entre llamadas:
    * "Mínima 1°C, viento sostenido 23 km/h."
    * "Frente frío con mínima de 1°C y viento de 23 km/h."
    * "Caída a 1°C con viento marcado (23 km/h)."
    * "Mínima nocturna 1°C; viento del sur a 23 km/h."
- NO uses comillas dobles, NO asteriscos, NO emojis.
- NO firmes, NO saludes.

NO HAGAS:
- "Temperatura 1°C + viento" — eso es la plantilla que estamos
  reemplazando; evitala.
- Inventar dirección del viento (sur/norte) si el input no lo dice.
- Agregar información que no esté en los datos.
- Respuestas conversacionales tipo "Las condiciones serán...".
- Redacciones largas o explicativas. UNA LÍNEA.

INCOHERENCIA CONCEPTUAL — NO MENCIONAR THI EN EVENTOS DE FRÍO:
El THI (índice temperatura-humedad) SOLO mide estrés calórico. NO
mide riesgo por frío. Si en el input el tipo de evento es "frio",
NO uses la palabra "THI" ni des un valor de THI aunque venga en los
datos. Hablá de: temperatura mínima, viento, humedad, sensación
térmica, barro. Para frío el indicador real es la combinación
T° baja + viento + humedad + agravantes — no un número THI.

DEVOLVÉ SOLO LA LÍNEA, sin texto adicional, sin markdown, sin guiones,
sin viñetas. Ejemplo de respuesta válida:
Mínima 1°C con viento sostenido de 23 km/h.
"""


def generar_resumen_operativo_llm(
    tipo: str,
    nivel: str,
    clima: Dict,
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    timeout_seg: int = 15,
) -> Optional[str]:
    """Reescribe la línea de condición climática del RESUMEN OPERATIVO
    para evitar la plantilla rígida. Conserva los datos numéricos.
    Devuelve None si la llamada falla."""
    key = _resolver_api_key(api_key)
    if not key:
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    partes_clima = []
    t = clima.get("temperatura") or clima.get("min_nocturna")
    if t is not None:
        partes_clima.append(f"T° {t}°C")
    v = clima.get("viento_kmh")
    if v:
        partes_clima.append(f"viento {v} km/h")
    ll = clima.get("lluvia_mm") or 0
    if ll:
        partes_clima.append(f"lluvia {ll} mm")
    if clima.get("barro"):
        partes_clima.append("barro presente")
    hr = clima.get("humedad_pct")
    if hr:
        partes_clima.append(f"HR {hr}%")
    # IMPORTANTE: el THI solo es un indicador válido para CALOR.
    # En eventos de FRÍO NO debe aparecer — es una incoherencia
    # conceptual (THI no mide estrés por frío). Solo lo incluimos
    # si el tipo de evento es calor.
    thi = clima.get("thi")
    if thi and (tipo or "").lower() == "calor":
        partes_clima.append(f"THI {thi}")

    if not partes_clima:
        return None

    clima_str = " · ".join(partes_clima)

    contexto = f"""DATOS DEL CLIMA del peor día del evento:
{clima_str}

Tipo de evento: {tipo}
Nivel del evento: {nivel}

Reescribí ahora la línea de condición climática (UNA SOLA LÍNEA, máx
22 palabras, conservando todos los datos numéricos). Devolvé SOLO la
línea, sin texto adicional."""

    try:
        client = Anthropic(api_key=key, timeout=float(timeout_seg))
        response = client.messages.create(
            model=model,
            max_tokens=120,
            system=SYSTEM_PROMPT_RESUMEN_OP,
            messages=[{"role": "user", "content": contexto}],
        )
        if not response.content:
            return None
        texto = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                texto += block.text
        texto = texto.strip()
        if not texto:
            return None

        # Sanitizar: quitar comillas, asteriscos, emojis comunes,
        # garantizar 1 sola línea.
        for ch in ['"', "'", "*", "_", "`"]:
            texto = texto.replace(ch, "")
        # Si tiene saltos de línea, agarrar solo la primera línea no vacía
        for linea in texto.split("\n"):
            if linea.strip():
                texto = linea.strip()
                break
        # Rechazar respuestas conversacionales/preguntas
        primera = texto[:40].lower()
        marcadores_invalidos = [
            "no puedo", "no tengo", "necesito", "podés", "podes",
            "¿", "?", "cuántos", "cuantos",
        ]
        if any(m in primera for m in marcadores_invalidos):
            return None
        # Largo razonable: ~30 palabras max (~180 chars)
        if len(texto) > 180:
            return None
        return texto
    except Exception as _e:
        _log.warning("LLM falló en %s: %s", "generar_resumen_operativo_llm", _e, exc_info=True)
        return None


def texto_a_html_parrafos(texto: str) -> str:
    """Convierte texto plano del LLM (con saltos de línea) en párrafos
    HTML simples. Hace escape básico de < y > para evitar inyección si
    el modelo devolviera algo raro."""
    if not texto:
        return ""
    txt = texto.replace("<", "&lt;").replace(">", "&gt;")
    parrafos = [p.strip() for p in txt.split("\n\n") if p.strip()]
    if not parrafos:
        # Sin doble salto: un solo párrafo
        return f"<p style='margin:6px 0 0; line-height:1.55;'>{txt}</p>"
    return "".join(
        f"<p style='margin:6px 0 0; line-height:1.55;'>{p}</p>"
        for p in parrafos
    )


# =====================================================================
# Inyectar zonas de confort + rigor + evidencia + espíritu en TODOS los
# system prompts. ESTA LÓGICA TIENE QUE ESTAR AL FINAL del archivo
# porque depende de que TODAS las constantes SYSTEM_PROMPT_* estén
# ya definidas. Si lo movés antes, NameError silencioso al import.
# =====================================================================
_EXTRAS_LLM = ("\n\n" + ZONAS_CONFORT_BOVINOS
               + "\n\n" + REGLAS_RIGOR_DATOS
               + "\n\n" + FUENTES_EVIDENCIA
               + "\n\n" + TONO_ASESOR_CAMPO
               + "\n\n" + ESPIRITU_HMS
               + "\n\n" + FILOSOFIA_EXPLICATIVA)
SYSTEM_PROMPT_ANALISIS = SYSTEM_PROMPT_ANALISIS + _EXTRAS_LLM
SYSTEM_PROMPT_UPDATE = SYSTEM_PROMPT_UPDATE + _EXTRAS_LLM
SYSTEM_PROMPT_DIARIO = SYSTEM_PROMPT_DIARIO + _EXTRAS_LLM
SYSTEM_PROMPT_WHATSAPP = SYSTEM_PROMPT_WHATSAPP + _EXTRAS_LLM
SYSTEM_PROMPT_ACCIONES = SYSTEM_PROMPT_ACCIONES + _EXTRAS_LLM
# El resumen operativo es solo variación de redacción del dato
# climático — le sumamos solo rigor (no espíritu ni zonas).
SYSTEM_PROMPT_RESUMEN_OP = (SYSTEM_PROMPT_RESUMEN_OP
                             + "\n\n" + REGLAS_RIGOR_DATOS)
