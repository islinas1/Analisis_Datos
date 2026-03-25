#  Análisis del Discurso Presidencial — ONU 2025

App interactiva en **Streamlit** para analizar discursos de la 80° Asamblea General de la ONU (septiembre 2025).

## Discursos analizados
| Presidente | País | Palabras | Perfil |
|---|---|---|---|
| Gabriel Boric | 🇨🇱 Chile | 2,593 | Izquierda progresista |
| Javier Milei | 🇦🇷 Argentina | 2,648 | Derecha libertaria |
| Yamandú Orsi | 🇺🇾 Uruguay | 1,669 | Centro moderado |

## Instalación

```bash

python -m venv venv
source venv/bin/activate 
# venv\Scripts\activate   


pip install -r requirements.txt


python -m spacy download es_core_news_md

streamlit run app.py
```

O simplemente:
```bash
bash setup.sh
streamlit run app.py
```

## Pipeline NLP (8 pasos)

1. **Limpieza** — Normalización, minúsculas, eliminación de puntuación
2. **Tokenización** — Segmentación + eliminación de stopwords (spaCy)
3. **Lematización / Stemming** — spaCy (lemas) + Snowball (stems)
4. **POS-Tagging** — Clasificación gramatical con `es_core_news_md`
5. **N-gramas** — Bigramas y trigramas más frecuentes
6. **Patrones sintácticos** — NOUN+VERB, NOUN+ADJ, ADJ+NOUN, VERB+NOUN
7. **Conteo de categorías** — Distribución porcentual de categorías POS
8. **Interpretación discursiva** — Análisis automático del perfil discursivo

## Estructura del proyecto

```
streamlit_app/
├── app.py                  # App principal de Streamlit
├── requirements.txt        # Dependencias Python
├── setup.sh               # Script de instalación rápida
├── README.md              # Este archivo
└── discursos/
    ├── boric.txt           # Discurso de Gabriel Boric (Chile)
    ├── milei.txt           # Discurso de Javier Milei (Argentina)
    └── orsi.txt            # Discurso de Yamandú Orsi (Uruguay)
```

## Visualizaciones incluidas

- 📊 Barras comparativas de categorías gramaticales
- 📊 Top 20 palabras más frecuentes (por presidente)
- ☁️ Nubes de palabras (por presidente)
- 📊 Bigramas y trigramas más frecuentes
- 🧩 Patrones sintácticos comparativos
- 🍩 Gráficos de dona (distribución POS)
- 🗺️ Heatmap comparativo
- 🎯 Radar de perfil discursivo

## Fuente
Compilación del CIEPS (Centro Internacional de Estudios Políticos y Sociales de Panamá)
