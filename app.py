import re, os, base64
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams


st.set_page_config(page_title="Análisis del Discurso — ONU 2025",
                   page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# ── Cargar CSS externo ────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def cargar_css(ruta):
    with open(ruta, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

cargar_css(os.path.join(APP_DIR, "estilos.css"))

# ── Helper: SVG como HTML img (base64) ────────────────────
def icono(nombre, clase="icon-inline"):
    """Devuelve tag <img> con el SVG en base64 para usar en markdown."""
    ruta = os.path.join(APP_DIR, "iconos", f"{nombre}.svg")
    if not os.path.exists(ruta):
        return ""
    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" class="{clase}">'

def icono_src(nombre):
    """Devuelve solo el src base64 para atributos img."""
    ruta = os.path.join(APP_DIR, "iconos", f"{nombre}.svg")
    if not os.path.exists(ruta):
        return ""
    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/svg+xml;base64,{b64}"


# ═══════════════════════════════════════════════════════════════
# MODELOS NLP (cacheados)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def cargar_modelos():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    return spacy.load('es_core_news_md'), SnowballStemmer('spanish'), set(stopwords.words('spanish'))

nlp, stemmer, stop_es = cargar_modelos()

COLORES = ['#2E86AB', '#A23B72', '#F18F01']
POS_ES = {
    'NOUN': 'Sustantivo', 'VERB': 'Verbo', 'ADJ': 'Adjetivo', 'ADV': 'Adverbio',
    'ADP': 'Preposición', 'DET': 'Determinante', 'PRON': 'Pronombre',
    'CCONJ': 'Conjunción', 'SCONJ': 'Subordinante', 'AUX': 'Auxiliar',
    'PROPN': 'Nombre propio', 'NUM': 'Número', 'PUNCT': 'Puntuación',
    'PART': 'Partícula', 'INTJ': 'Interjección', 'X': 'Otro',
    'SYM': 'Símbolo', 'SPACE': 'Espacio',
}

# ═══════════════════════════════════════════════════════════════
# CORPUS
# ═══════════════════════════════════════════════════════════════

def cargar_discurso(archivo, fallback=""):
    ruta = os.path.join(APP_DIR, "discursos", archivo)
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    return fallback

DISCURSOS = {
    "Gabriel Boric (Chile)": cargar_discurso("boric.txt", "Discurso no encontrado."),
    "Javier Milei (Argentina)": cargar_discurso("milei.txt", "Discurso no encontrado."),
    "Yamandú Orsi (Uruguay)": cargar_discurso("orsi.txt", "Discurso no encontrado."),
}

# ═══════════════════════════════════════════════════════════════
# FUNCIONES NLP
# ═══════════════════════════════════════════════════════════════

def limpiar_texto(t):
    t = t.lower(); t = re.sub(r'\n+', ' ', t); t = re.sub(r'\s+', ' ', t)
    return re.sub(r'[^\w\sáéíóúüñ¿¡]', '', t).strip()

def tokenizar(t):
    return [tk.text for tk in nlp(t) if tk.text not in stop_es and len(tk.text) > 1 and not tk.is_punct]

def lematizar_y_stem(t):
    return [{'Token': tk.text, 'Lema': tk.lemma_, 'Stem': stemmer.stem(tk.text)}
            for tk in nlp(t) if tk.text not in stop_es and len(tk.text) > 1 and not tk.is_punct]

def pos_tagging(t):
    return [{'Token': tk.text, 'POS': tk.pos_, 'POS_es': POS_ES.get(tk.pos_, tk.pos_), 'Tag': tk.tag_}
            for tk in nlp(t) if not tk.is_space]

def generar_ngramas(tokens, n=2, top=15):
    return Counter(list(ngrams(tokens, n))).most_common(top)

def identificar_patrones(t):
    doc = nlp(t)
    pat = {'NOUN + VERB': [], 'NOUN + ADJ': [], 'ADJ + NOUN': [], 'VERB + NOUN': []}
    for i in range(len(doc) - 1):
        clave = f'{doc[i].pos_} + {doc[i+1].pos_}'
        if clave in pat: pat[clave].append(f'{doc[i].text} {doc[i+1].text}')
    return pat

def contar_categorias(pos_tags):
    v = [t for t in pos_tags if t['POS'] not in ('SPACE', 'PUNCT')]
    total = len(v); conteo = Counter(t['POS'] for t in v)
    return {p: {'categoría': POS_ES.get(p, p), 'frecuencia': f, 'porcentaje': round(f/total*100, 2)}
            for p, f in conteo.most_common()}

def interpretar(nombre, cat, pat, bg):
    pct = {v['categoría']: v['porcentaje'] for v in cat.values()}
    s, v, a, p = pct.get('Sustantivo',0), pct.get('Verbo',0), pct.get('Adjetivo',0), pct.get('Pronombre',0)
    r = {}
    r['orientacion'] = f'NOMINAL (sustantivos {s}% vs verbos {v}%)' if s > v else f'VERBAL (verbos {v}% vs sustantivos {s}%)'
    r['orientacion_desc'] = 'Centrado en conceptos e ideas abstractas.' if s > v else 'Orientado a la acción y transformación.'
    r['adjetival'] = f'Alta carga adjetival ({a}%): componente evaluativo.' if a > 10 else (f'Carga moderada ({a}%): equilibrio.' if a > 5 else f'Baja carga ({a}%): directo y factual.')
    r['pronominal'] = f'Uso pronominal: {p}%' + (' — cercanía e inclusión.' if p > 5 else '.')
    nv, na, vn = len(pat.get('NOUN + VERB',[])), len(pat.get('NOUN + ADJ',[])), len(pat.get('VERB + NOUN',[]))
    mejor = max([('SUST+VERB',nv),('SUST+ADJ',na),('VERB+SUST',vn)], key=lambda x: x[1])
    r['patron'] = f'{mejor[0]} ({mejor[1]} casos)'
    r['temas'] = [' '.join(b) for b, _ in bg[:5]] if bg else []
    return r

# ═══════════════════════════════════════════════════════════════
# PROCESAMIENTO (cacheado)
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def procesar_discursos(disc):
    res = {}
    for nombre, crudo in disc.items():
        corto = nombre.split('(')[0].strip()
        limpio = limpiar_texto(crudo)
        tokens = tokenizar(limpio)
        lemas = lematizar_y_stem(limpio)
        pos = pos_tagging(limpio)
        bgs = generar_ngramas(tokens, 2, 15)
        tgs = generar_ngramas(tokens, 3, 10)
        pats = identificar_patrones(limpio)
        cats = contar_categorias(pos)
        interp = interpretar(nombre, cats, pats, bgs)
        res[nombre] = dict(corto=corto, texto_crudo=crudo, texto_limpio=limpio,
                           tokens=tokens, lemas=lemas, pos=pos, bigramas=bgs,
                           trigramas=tgs, patrones=pats, categorias=cats,
                           interpretacion=interp, n_palabras_orig=len(crudo.split()),
                           n_palabras_limpio=len(limpio.split()), n_tokens=len(tokens))
    return res

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

ICONOS_NAV = {
    "Presentación (Diapositivas)": "presentacion",
    "Resumen general": "home",
    "1. Limpieza": "limpieza",
    "2. Tokenización": "tokenizar",
    "3. Lematización / Stemming": "libro",
    "4. POS-Tagging": "etiqueta",
    "5. N-gramas": "enlace",
    "6. Patrones sintácticos": "puzzle",
    "7. Conteo de categorías": "grafico",
    "8. Interpretación discursiva": "cerebro",
}

with st.sidebar:
    st.markdown(f'{icono("microfono", "icon-sidebar")}', unsafe_allow_html=True)
    st.markdown("# Análisis del Discurso")
    st.markdown("**80° Asamblea General ONU**<br>Septiembre 2025", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Presidentes")
    st.markdown(f'<span class="president-tag tag-boric">{icono("chile","icon-flag")} Gabriel Boric</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="president-tag tag-milei">{icono("argentina","icon-flag")} Javier Milei</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="president-tag tag-orsi">{icono("uruguay","icon-flag")} Yamandú Orsi</span>', unsafe_allow_html=True)
    st.divider()

    opciones_nav = list(ICONOS_NAV.keys())
    seccion = st.radio("Navegación", opciones_nav, label_visibility="collapsed")

    st.divider()
    st.caption("Pipeline: limpieza → tokenización → lematización → POS-tagging → n-gramas → patrones → conteo → interpretación")


# ═══════════════════════════════════════════════════════════════
# PROCESAR DATOS
# ═══════════════════════════════════════════════════════════════

with st.spinner("Procesando discursos con spaCy..."):
    datos = procesar_discursos(DISCURSOS)

NOMBRES = list(DISCURSOS.keys())
CORTOS = [datos[n]['corto'] for n in NOMBRES]
BANDERAS = ['chile', 'argentina', 'uruguay']


# ═══════════════════════════════════════════════════════════════
# HELPER: Step header con icono
# ═══════════════════════════════════════════════════════════════

def step_header(icon_name, titulo):
    st.markdown(f'''<div class="step-header">
        <img src="{icono_src(icon_name)}"> <h3>{titulo}</h3>
    </div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECCIÓN: PRESENTACIÓN (DIAPOSITIVAS)
# ═══════════════════════════════════════════════════════════════

if "Presentación" in seccion:
    total_slides = 6
    if 'slide_num' not in st.session_state:
        st.session_state.slide_num = 1

    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        if st.button("Anterior", use_container_width=True, disabled=st.session_state.slide_num <= 1):
            st.session_state.slide_num -= 1; st.rerun()
    with nav_cols[1]:
        st.markdown(f"<p style='text-align:center; color:#8892b0;'>Diapositiva {st.session_state.slide_num} de {total_slides}</p>", unsafe_allow_html=True)
    with nav_cols[2]:
        if st.button("Siguiente", use_container_width=True, disabled=st.session_state.slide_num >= total_slides):
            st.session_state.slide_num += 1; st.rerun()

    sn = st.session_state.slide_num

    if sn == 1:
        st.markdown(f"""
        <div class="slide slide-cover">
            <img src="{icono_src('microfono')}" class="cover-icon">
            <p class="subtitle">Universidad Mayor de San Andrés</p>
            <h1>Análisis del Discurso Presidencial</h1>
            <h2>Procesamiento de Lenguaje Natural aplicado a discursos<br>de la 80° Asamblea General — ONU 2025</h2>
            <div class="authors">
                <div class="author-row"><img src="{icono_src('persona')}"><p>Ian Salinas</p></div>
                <div class="author-row"><img src="{icono_src('persona')}"><p>Ramiro Mamani</p></div>
            </div>
            <span class="course-badge">Procesamiento del Lenguaje Natural (PLN)</span>
            <p class="date-label">2025</p>
            <span class="slide-counter">1 / {total_slides}</span>
        </div>""", unsafe_allow_html=True)

    elif sn == 2:
        st.markdown(f"""
        <div class="slide slide-q">
            <p class="q-number">Pregunta 01</p>
            <h2>¿Qué tipos de palabras predominan en los discursos políticos?</h2>
            <div class="answer">
                <p>En los tres discursos analizados (Boric, Milei, Orsi), el patrón es consistente:
                los <span class="key">sustantivos</span> dominan ampliamente (<span class="accent">30–38%</span>),
                reflejando una orientación <span class="key">nominal</span> centrada en conceptos abstractos
                (democracia, libertad, humanidad, paz).</p>
                <div class="finding-card">
                    <p class="label">Hallazgo clave</p>
                    <p>Los <span class="key">verbos</span> (~15-18%) van segundo, seguidos de
                    <span class="key">determinantes</span> y <span class="key">preposiciones</span>.
                    Los adjetivos varían según estilo retórico.</p>
                </div>
                <div class="finding-card">
                    <p class="label">Implicación</p>
                    <p>Los discursos ante la ONU priorizan <span class="accent">definir realidades</span>
                    antes que <span class="accent">proponer acciones</span>. Es un discurso más declarativo que performativo.</p>
                </div>
            </div>
            <span class="slide-counter">2 / {total_slides}</span>
        </div>""", unsafe_allow_html=True)
        cats_interes = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'DET']
        regs = [{'Presidente': datos[n]['corto'], 'Categoría': POS_ES[c], '%': datos[n]['categorias'].get(c, {}).get('porcentaje', 0)}
                for n in NOMBRES for c in cats_interes]
        fig = px.bar(pd.DataFrame(regs), x='Categoría', y='%', color='Presidente', barmode='group',
                     color_discrete_sequence=COLORES, title='Distribución de categorías gramaticales')
        fig.update_layout(height=420, template='plotly_dark', yaxis_title='Porcentaje (%)')
        st.plotly_chart(fig, use_container_width=True)

    elif sn == 3:
        st.markdown(f"""
        <div class="slide slide-q">
            <p class="q-number">Pregunta 02</p>
            <h2>¿Qué tipo de verbos aparecen con mayor frecuencia?</h2>
            <div class="answer">
                <div class="finding-card"><p class="label">1. Verbos de estado</p>
                    <p><span class="accent">ser, estar, haber</span> — construyen la realidad discursiva.</p></div>
                <div class="finding-card"><p class="label">2. Verbos volitivos</p>
                    <p><span class="accent">querer, deber, poder, hacer</span> — expresan voluntad política.</p></div>
                <div class="finding-card"><p class="label">3. Verbos declarativos</p>
                    <p><span class="accent">decir, anunciar, reclamar</span> — ejecutan acciones desde el habla.</p></div>
                <p style="margin-top:1rem;">Milei privilegia verbos de acción concreta, Boric verbos emocionales y morales,
                Orsi verbos de posición diplomática.</p>
            </div>
            <span class="slide-counter">3 / {total_slides}</span>
        </div>""", unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=3, subplot_titles=[datos[n]['corto'] for n in NOMBRES])
        for i, nombre in enumerate(NOMBRES):
            vf = Counter(t['Token'] for t in datos[nombre]['pos'] if t['POS'] == 'VERB' and len(t['Token']) > 2).most_common(12)
            if vf:
                w, c = zip(*vf)
                fig.add_trace(go.Bar(y=list(w)[::-1], x=list(c)[::-1], orientation='h',
                                     marker_color=COLORES[i], showlegend=False), row=1, col=i+1)
        fig.update_layout(height=420, template='plotly_dark', title='Verbos más frecuentes por presidente')
        st.plotly_chart(fig, use_container_width=True)

    elif sn == 4:
        st.markdown(f"""
        <div class="slide slide-q">
            <p class="q-number">Pregunta 03 — Textos académicos</p>
            <h2>¿Qué patrones lingüísticos permiten detectar conceptos científicos en textos académicos?</h2>
            <div class="answer">
                <p>En textos de <span class="accent">IA, Lingüística Computacional y PLN</span>:</p>
                <div class="finding-card"><p class="label">Patrón 1: SUSTANTIVO + ADJETIVO técnico</p>
                    <p><span class="accent">«red neuronal»</span>, <span class="accent">«aprendizaje profundo»</span>,
                    <span class="accent">«análisis semántico»</span></p></div>
                <div class="finding-card"><p class="label">Patrón 2: SUST + PREP + SUST</p>
                    <p><span class="accent">«procesamiento de lenguaje»</span>, <span class="accent">«modelo de atención»</span></p></div>
                <div class="finding-card"><p class="label">Patrón 3: Voz pasiva + SUST</p>
                    <p><span class="accent">«fue entrenado»</span>, <span class="accent">«se implementó»</span></p></div>
                <div class="finding-card"><p class="label">Patrón 4: Alta densidad SUST + baja PRON</p>
                    <p>La ausencia de «nosotros», «yo» es marcador de registro científico vs. político.</p></div>
            </div>
            <span class="slide-counter">4 / {total_slides}</span>
        </div>""", unsafe_allow_html=True)
        cats_c = ['Sustantivo', 'Verbo', 'Adjetivo', 'Pronombre', 'Preposición']
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Discurso político (ONU)', x=cats_c, y=[35,17,8,7,11], marker_color='#2E86AB'))
        fig.add_trace(go.Bar(name='Texto académico (IA/PLN)', x=cats_c, y=[42,12,11,2,15], marker_color='#64ffda'))
        fig.update_layout(barmode='group', height=400, template='plotly_dark',
                          title='Distribución POS: político vs académico', yaxis_title='%')
        st.plotly_chart(fig, use_container_width=True)

    elif sn == 5:
        st.markdown(f"""
        <div class="slide slide-q">
            <p class="q-number">Pregunta 04 — Informativo vs. opinión</p>
            <h2>¿Existen diferencias en el uso de verbos, sustantivos y pronombres en textos informativos vs textos de opinión?</h2>
            <div class="answer">
                <p>Sí, las diferencias son <span class="key">sistemáticas y medibles</span>:</p>
                <div class="finding-card"><p class="label">Sustantivos</p>
                    <p><span class="accent">Informativos:</span> sustantivos concretos y propios (países, cifras).<br>
                    <span class="accent">Opinión:</span> sustantivos abstractos (justicia, dignidad, humanidad).</p></div>
                <div class="finding-card"><p class="label">Verbos</p>
                    <p><span class="accent">Informativos:</span> indicativo pasado (reportar hechos).<br>
                    <span class="accent">Opinión:</span> subjuntivo/imperativo (deseo, obligación).</p></div>
                <div class="finding-card"><p class="label">Pronombres</p>
                    <p><span class="accent">Informativos:</span> uso reducido, tercera persona.<br>
                    <span class="accent">Opinión:</span> alto uso de «nosotros» (inclusión), «yo» (autoridad).</p></div>
            </div>
            <span class="slide-counter">5 / {total_slides}</span>
        </div>""", unsafe_allow_html=True)
        cats = ['Sust.\nconcretos','Sust.\nabstractos','Verbos\nindicativo','Verbos\nsubjuntivo','Pronombres\npersonales','Adj.\ncalificativos']
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Informativo', x=cats, y=[38,12,22,3,4,5], marker_color='#2E86AB'))
        fig.add_trace(go.Bar(name='Opinión', x=cats, y=[15,35,10,14,12,14], marker_color='#F18F01'))
        fig.update_layout(barmode='group', height=420, template='plotly_dark',
                          title='Informativo vs opinión (modelo teórico)', yaxis_title='%')
        st.plotly_chart(fig, use_container_width=True)

    elif sn == 6:
        st.markdown(f"""
        <div class="slide slide-cover">
            <img src="{icono_src('cerebro')}" class="cover-icon">
            <h1 style="font-size:2rem;">Conclusiones</h1>
            <h2 style="font-size:1.1rem; max-width:700px;">El PLN permite revelar estructuras ocultas en el discurso político,
            diferenciando estilos retóricos, detectando patrones gramaticales y
            cuantificando la orientación ideológica a través del lenguaje.</h2>
            <div class="authors" style="margin-top:2rem;">
                <p style="color:#64ffda; font-size:0.9rem; font-weight:600;">Herramientas utilizadas</p>
                <p style="font-size:0.95rem;">spaCy · NLTK · Streamlit · Plotly · WordCloud</p>
                <p style="color:#64ffda; font-size:0.9rem; font-weight:600; margin-top:0.8rem;">Corpus analizado</p>
                <p style="font-size:0.95rem;">3 discursos — 80° Asamblea General ONU (sept. 2025)</p>
                <p style="font-size:0.95rem;">Boric (Chile) · Milei (Argentina) · Orsi (Uruguay)</p>
            </div>
            <div style="margin-top:2rem;">
                <p class="subtitle">Ian Salinas · Ramiro Mamani</p>
                <span class="course-badge">Procesamiento del Lenguaje Natural (PLN)</span>
            </div>
            <span class="slide-counter">6 / {total_slides}</span>
        </div>""", unsafe_allow_html=True)

    st.progress(sn / total_slides)
    st.markdown("---")
    names = ["1. Carátula", "2. Palabras predominantes", "3. Verbos frecuentes",
             "4. Patrones académicos", "5. Informativo vs opinión", "6. Conclusiones"]
    sel_s = st.selectbox("Ir a diapositiva:", names, index=st.session_state.slide_num - 1, key="sj")
    nn = names.index(sel_s) + 1
    if nn != st.session_state.slide_num:
        st.session_state.slide_num = nn; st.rerun()


# ═══════════════════════════════════════════════════════════════
# SECCIÓN: RESUMEN GENERAL
# ═══════════════════════════════════════════════════════════════

elif "Resumen" in seccion:
    st.markdown(f"# {icono('microfono','icon-inline-lg')} Análisis del Discurso Presidencial", unsafe_allow_html=True)
    st.markdown("### 80° Asamblea General de las Naciones Unidas — Septiembre 2025")
    st.markdown("---")
    cols = st.columns(3)
    for i, nombre in enumerate(NOMBRES):
        d = datos[nombre]
        with cols[i]:
            st.markdown(f"""<div class="metric-card">
                <h3 style="color:{COLORES[i]}">{icono(BANDERAS[i],'icon-flag')} {d['corto']}</h3>
                <p>{d['n_palabras_orig']:,}</p><h3>palabras</h3>
            </div>""", unsafe_allow_html=True)
            st.metric("Tokens útiles", d['n_tokens'])
    st.markdown("---")
    st.markdown(f"### {icono('scroll')} Vista previa de los discursos", unsafe_allow_html=True)
    tabs = st.tabs([datos[n]['corto'] for n in NOMBRES])
    for i, nombre in enumerate(NOMBRES):
        with tabs[i]:
            st.text_area("", DISCURSOS[nombre][:2000] + "\n\n[...]", height=300, key=f"pv_{i}", disabled=True)


# ═══════════════════════════════════════════════════════════════
# PASO 1: LIMPIEZA
# ═══════════════════════════════════════════════════════════════

elif "Limpieza" in seccion:
    step_header("limpieza", "Paso 1: Limpieza de texto")
    st.markdown("Normalización a minúsculas, eliminación de saltos de línea, puntuación y caracteres especiales.")
    filas = [{'Presidente': datos[n]['corto'], 'Palabras originales': datos[n]['n_palabras_orig'],
              'Palabras limpias': datos[n]['n_palabras_limpio'],
              'Vista previa': datos[n]['texto_limpio'][:120] + '…'} for n in NOMBRES]
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    sel = st.selectbox("Ver texto limpio completo:", CORTOS, key="l_s")
    st.code(datos[NOMBRES[CORTOS.index(sel)]]['texto_limpio'][:1000] + "\n\n[...]", language=None)


# ═══════════════════════════════════════════════════════════════
# PASO 2: TOKENIZACIÓN
# ═══════════════════════════════════════════════════════════════

elif "Tokenización" in seccion:
    step_header("tokenizar", "Paso 2: Tokenización")
    filas = [{'Presidente': datos[n]['corto'], 'Tokens totales': datos[n]['n_palabras_limpio'],
              'Tokens útiles': datos[n]['n_tokens'],
              'Reducción': f"{100 - round(datos[n]['n_tokens']/datos[n]['n_palabras_limpio']*100)}%",
              'Primeros 10': ', '.join(datos[n]['tokens'][:10])} for n in NOMBRES]
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    fig = go.Figure()
    for i, n in enumerate(NOMBRES):
        fig.add_trace(go.Bar(name=datos[n]['corto'], x=['Tokens totales', 'Tokens útiles'],
                             y=[datos[n]['n_palabras_limpio'], datos[n]['n_tokens']], marker_color=COLORES[i]))
    fig.update_layout(barmode='group', title='Tokens totales vs útiles', height=400, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PASO 3: LEMATIZACIÓN
# ═══════════════════════════════════════════════════════════════

elif "Lematización" in seccion:
    step_header("libro", "Paso 3: Lematización y Stemming")
    sel = st.selectbox("Seleccionar presidente:", CORTOS, key="lem_s")
    n_rows = st.slider("Filas", 10, 50, 25, key="lem_sl")
    st.dataframe(pd.DataFrame(datos[NOMBRES[CORTOS.index(sel)]]['lemas'][:n_rows]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PASO 4: POS-TAGGING
# ═══════════════════════════════════════════════════════════════

elif "POS-Tagging" in seccion:
    step_header("etiqueta", "Paso 4: POS-Tagging")
    sel = st.selectbox("Seleccionar presidente:", CORTOS, key="pos_s")
    n_sel = NOMBRES[CORTOS.index(sel)]
    n_rows = st.slider("Filas", 10, 50, 25, key="pos_sl")
    st.dataframe(pd.DataFrame(datos[n_sel]['pos'][:n_rows]), use_container_width=True, hide_index=True)
    pos_data = datos[n_sel]['categorias']
    fig = px.pie(names=[v['categoría'] for v in pos_data.values()],
                 values=[v['frecuencia'] for v in pos_data.values()],
                 title=f'Distribución POS — {sel}', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=450, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PASO 5: N-GRAMAS
# ═══════════════════════════════════════════════════════════════

elif "N-gramas" in seccion:
    step_header("enlace", "Paso 5: N-gramas")
    tab_bi, tab_tri, tab_nube = st.tabs(["Bigramas", "Trigramas", "Nubes de palabras"])
    with tab_bi:
        cols = st.columns(3)
        for i, nombre in enumerate(NOMBRES):
            with cols[i]:
                bgs = datos[nombre]['bigramas'][:12]
                et = [' '.join(bg) for bg, _ in bgs]; vl = [f for _, f in bgs]
                fig = px.bar(x=vl[::-1], y=et[::-1], orientation='h', title=datos[nombre]['corto'],
                             color_discrete_sequence=[COLORES[i]])
                fig.update_layout(height=450, template='plotly_dark', xaxis_title='Frecuencia', yaxis_title='', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    with tab_tri:
        for nombre in NOMBRES:
            tris = datos[nombre]['trigramas'][:8]
            if tris:
                st.markdown(f"**{datos[nombre]['corto']}**")
                st.dataframe(pd.DataFrame([{'Trigrama': ' '.join(t), 'Freq': f} for t, f in tris]),
                             use_container_width=True, hide_index=True)
    with tab_nube:
        cmaps = ['viridis', 'plasma', 'inferno']
        cols = st.columns(3)
        for i, nombre in enumerate(NOMBRES):
            with cols[i]:
                wc = WordCloud(width=600, height=350, background_color='#0a192f',
                               colormap=cmaps[i], max_words=70).generate(' '.join(datos[nombre]['tokens']))
                fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor('#0a192f')
                ax.imshow(wc, interpolation='bilinear')
                ax.set_title(datos[nombre]['corto'], color='white', fontsize=13, fontweight='bold'); ax.axis('off')
                st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# PASO 6: PATRONES
# ═══════════════════════════════════════════════════════════════

elif "Patrones" in seccion:
    step_header("puzzle", "Paso 6: Patrones sintácticos")
    regs = [{'Presidente': datos[n]['corto'], 'Patrón': p, 'Ocurrencias': len(e)}
            for n in NOMBRES for p, e in datos[n]['patrones'].items()]
    fig = px.bar(pd.DataFrame(regs), x='Patrón', y='Ocurrencias', color='Presidente', barmode='group',
                 color_discrete_sequence=COLORES, title='Patrones sintácticos por discurso')
    fig.update_layout(height=450, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    sel = st.selectbox("Ver ejemplos de:", CORTOS, key="pat_s")
    for p, ej in datos[NOMBRES[CORTOS.index(sel)]]['patrones'].items():
        if ej:
            with st.expander(f"**{p}** — {len(ej)} ocurrencias"):
                st.write(", ".join(ej[:20]))


# ═══════════════════════════════════════════════════════════════
# PASO 7: CONTEO
# ═══════════════════════════════════════════════════════════════

elif "Conteo" in seccion:
    step_header("grafico", "Paso 7: Conteo de categorías gramaticales")
    tab_b, tab_h, tab_r = st.tabs(["Barras comparativas", "Heatmap", "Radar"])
    cats_i = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'DET']
    with tab_b:
        regs = [{'Presidente': datos[n]['corto'], 'Categoría': POS_ES[c], '%': datos[n]['categorias'].get(c,{}).get('porcentaje',0)}
                for n in NOMBRES for c in cats_i]
        fig = px.bar(pd.DataFrame(regs), x='Categoría', y='%', color='Presidente', barmode='group',
                     color_discrete_sequence=COLORES, title='Distribución de categorías gramaticales')
        fig.update_layout(height=500, template='plotly_dark', yaxis_title='%')
        st.plotly_chart(fig, use_container_width=True)
    with tab_h:
        dh = {datos[n]['corto']: {POS_ES[c]: datos[n]['categorias'].get(c,{}).get('porcentaje',0) for c in cats_i} for n in NOMBRES}
        df_hm = pd.DataFrame(dh).T
        fig, ax = plt.subplots(figsize=(12, 4)); fig.patch.set_facecolor('#0a192f'); ax.set_facecolor('#0a192f')
        sns.heatmap(df_hm, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=2, linecolor='#0a192f', cbar_kws={'label': '%'}, ax=ax)
        ax.set_title('Mapa de calor — categorías gramaticales (%)', fontsize=14, fontweight='bold', color='white', pad=15)
        ax.tick_params(colors='white'); plt.setp(ax.get_xticklabels(), color='white'); plt.setp(ax.get_yticklabels(), color='white')
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with tab_r:
        cr = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']; cr_es = [POS_ES[c] for c in cr]
        fig = go.Figure()
        for i, n in enumerate(NOMBRES):
            vals = [datos[n]['categorias'].get(c,{}).get('porcentaje',0) for c in cr]; vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(r=vals, theta=cr_es+[cr_es[0]], fill='toself',
                                          name=datos[n]['corto'], line_color=COLORES[i], opacity=0.7))
        fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)')),
                          title='Perfil discursivo — radar', height=550, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PASO 8: INTERPRETACIÓN
# ═══════════════════════════════════════════════════════════════

elif "Interpretación" in seccion:
    step_header("cerebro", "Paso 8: Interpretación discursiva")
    for i, nombre in enumerate(NOMBRES):
        d = datos[nombre]; interp = d['interpretacion']
        temas = ', '.join(f'«{t}»' for t in interp.get('temas', []))
        st.markdown(f"""<div class="interpretation-box">
            <h4><img src="{icono_src(BANDERAS[i])}"> {d['corto']}</h4>
            <p>► Discurso <span class="highlight">{interp['orientacion']}</span></p>
            <p style="font-size:0.9rem; margin-left:1.5rem;">{interp['orientacion_desc']}</p>
            <p>► {interp['adjetival']}</p>
            <p>► {interp['pronominal']}</p>
            <p>► Patrón dominante: <span class="highlight">{interp['patron']}</span></p>
            <p>► Ejes temáticos: <span class="highlight">{temas}</span></p>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"### {icono('navegacion')} Resumen comparativo", unsafe_allow_html=True)
    resumen = [{'Presidente': datos[n]['corto'], 'Palabras': datos[n]['n_palabras_orig'],
                'Tokens útiles': datos[n]['n_tokens'],
                'Orientación': datos[n]['interpretacion']['orientacion'].split('(')[0].strip(),
                'Patrón': datos[n]['interpretacion']['patron']} for n in NOMBRES]
    st.dataframe(pd.DataFrame(resumen), use_container_width=True, hide_index=True)
    st.caption("Fuente: Compilación CIEPS — 80° Asamblea General ONU, septiembre 2025")
