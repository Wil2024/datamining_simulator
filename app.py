import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Data Mining Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .main-header p  { font-size: 1rem; color: #a0aec0; margin: 0.5rem 0 0; }

    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .kpi-card h3 { font-size: 2rem; font-weight: 800; margin: 0; }
    .kpi-card p  { font-size: 0.85rem; margin: 0; opacity: 0.85; }

    .insight-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #0c4a6e;
    }
    .warning-box {
        background-color: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #7c2d12;
    }
    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #14532d;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #2d6a9f);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Simulador de Minería de Datos</h1>
    <p>Análisis Avanzado de Datos · Empresas Peruanas</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def download_btn(df, filename, label):
    data = to_excel_bytes(df)
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" style="background:#0ea5e9;color:white;padding:8px 16px;border-radius:6px;text-decoration:none;font-weight:600;">⬇️ {label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 <b>Interpretación:</b> {text}</div>', unsafe_allow_html=True)

def warning_box(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

def success_box(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

PALETA = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7", "#3a0ca3", "#560bad", "#480ca8"]

# ─────────────────────────────────────────────
# SIDEBAR — CARGA DE ARCHIVOS
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.markdown("## 📁 Carga de Datos")

    empresa_nombre = st.text_input("Nombre de la empresa", placeholder="Ej: AJE Delivery, InkaSalud...")

    uploaded_trans = st.file_uploader("📦 Transacciones (.xlsx)", type=["xlsx"])
    uploaded_rev   = st.file_uploader("⭐ Reseñas (.xlsx)", type=["xlsx"])
    uploaded_cli   = st.file_uploader("👥 Clientes (.xlsx) [Opcional]", type=["xlsx"])

    st.markdown("---")
    st.markdown("### ℹ️ Columnas requeridas")
    st.markdown("""
**Transacciones:**
- `customer_id`, `order_id`, `order_date`
- `product_id`, `product_name`, `category`
- `price`, `quantity`, `total_amount`
- `place`, `email`

**Reseñas:**
- `customer_id`, `order_id`
- `review_text`, `rating`, `sentiment`
""")
    st.markdown("---")
    st.caption("© 2026 Data Mining Simulator")

# ─────────────────────────────────────────────
# MAIN — VALIDACIÓN Y CARGA
# ─────────────────────────────────────────────
REQUIRED_TRANS = {"customer_id","order_id","order_date","product_id","product_name",
                  "category","price","quantity","total_amount","place","email"}
REQUIRED_REV   = {"customer_id","order_id","review_text","rating","sentiment"}

if not uploaded_trans or not uploaded_rev:
    st.markdown("""
    <div class="warning-box">
    <b>Bienvenido al Simulador MBA de Minería de Datos.</b><br>
    Carga los archivos de <b>Transacciones</b> y <b>Reseñas</b> desde el panel izquierdo para comenzar.<br><br>
    Empresas disponibles con datasets listos:<br>
    🛒 <b>AJE Delivery</b> (Bebidas/Abarrotes - Lima)<br>
    💊 <b>InkaSalud</b> (Farmacia - Lima)<br>
    💻 <b>TecnoExpress</b> (Electrónica - Arequipa)<br>
    👗 <b>Moda Killa</b> (Moda/Ropa - Trujillo)
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Carga ───
try:
    df_t = pd.read_excel(uploaded_trans)
    df_r = pd.read_excel(uploaded_rev)
except Exception as e:
    st.error(f"Error al leer los archivos: {e}")
    st.stop()

# ─── Validar columnas ───
missing_t = REQUIRED_TRANS - set(df_t.columns)
missing_r = REQUIRED_REV   - set(df_r.columns)
if missing_t or missing_r:
    if missing_t:
        st.error(f"❌ Faltan columnas en Transacciones: {missing_t}")
    if missing_r:
        st.error(f"❌ Faltan columnas en Reseñas: {missing_r}")
    st.stop()

df_c = pd.read_excel(uploaded_cli) if uploaded_cli else None
nombre_empresa = empresa_nombre.strip() if empresa_nombre.strip() else "Mi Empresa"

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab_overview, tab_cluster, tab_sentiment, tab_apriori, tab_geo = st.tabs([
    "📊 Vista General",
    "🎯 Segmentación (K-Means)",
    "💬 Sentimiento (Naive Bayes)",
    "🔗 Asociación (Apriori)",
    "🗺️ Análisis Geográfico",
])

# ══════════════════════════════════════════════
# TAB 1 — VISTA GENERAL
# ══════════════════════════════════════════════
with tab_overview:
    st.markdown(f'<div class="section-header">📊 Dashboard General — {nombre_empresa}</div>', unsafe_allow_html=True)

    # KPIs
    total_ventas   = df_t["total_amount"].sum()
    total_ordenes  = df_t["order_id"].nunique()
    total_clientes = df_t["customer_id"].nunique()
    ticket_prom    = df_t.groupby("order_id")["total_amount"].sum().mean()
    avg_rating     = df_r["rating"].mean()
    pct_positivo   = (df_r["sentiment"] == "Positivo").mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, lbl in [
        (c1, f"S/ {total_ventas:,.0f}", "Ventas Totales"),
        (c2, f"{total_ordenes:,}", "Total Órdenes"),
        (c3, f"{total_clientes:,}", "Clientes Únicos"),
        (c4, f"S/ {ticket_prom:.1f}", "Ticket Promedio"),
        (c5, f"{avg_rating:.2f} ⭐", "Rating Promedio"),
        (c6, f"{pct_positivo:.1f}%", "Reseñas Positivas"),
    ]:
        col.markdown(f'<div class="kpi-card"><h3>{val}</h3><p>{lbl}</p></div>', unsafe_allow_html=True)

   # st.markdown("---")
    #insight("Los gráficos de análisis descriptivo (Ventas por categoría, #Evolución temporal, Distribución de Ratings y Sentimiento) han sido #deshabilitados. Como parte de tu desafío MBA, debes descargar los datos y #construir estas visualizaciones por tu cuenta.")
    
    st.markdown("---")
    st.subheader("🔍 Vista Previa de Datos")
    col_e, col_f = st.columns(2)
    with col_e:
        st.write(f"**Transacciones** ({df_t.shape[0]:,} filas)")
        st.dataframe(df_t.head(8), use_container_width=True)
    with col_f:
        st.write(f"**Reseñas** ({df_r.shape[0]:,} filas)")
        st.dataframe(df_r.head(8), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — CLUSTERING K-MEANS
# ══════════════════════════════════════════════
with tab_cluster:
    st.markdown('<div class="section-header">🎯 Segmentación de Clientes — K-Means Clustering</div>', unsafe_allow_html=True)

    st.markdown("""
    **¿Qué hace?** Agrupa automáticamente a los clientes según su comportamiento de compra.
    Cada grupo (clúster) comparte características similares: cuánto gastan, con qué frecuencia compran y qué variedad de productos eligen.
    """)

    with st.expander("⚙️ Parámetros del Modelo", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Número de clústeres (K):", 2, 8, 3,
                help="Cuántos grupos de clientes quieres identificar. 3-4 es lo recomendado para empezar.")
        with col2:
            features_sel = st.multiselect(
                "Variables para el clustering:",
                ["total_spent", "avg_spent", "order_count", "unique_categories"],
                default=["total_spent", "avg_spent", "order_count", "unique_categories"],
                help="Las variables que el algoritmo usará para agrupar clientes."
            )

    if st.button("▶️ Ejecutar Segmentación K-Means", type="primary"):
        if len(features_sel) < 2:
            warning_box("Selecciona al menos 2 variables para el clustering.")
        else:
            with st.spinner("Entrenando modelo K-Means..."):
                feat = df_t.groupby("customer_id").agg(
                    total_spent=("total_amount", "sum"),
                    avg_spent=("total_amount", "mean"),
                    order_count=("order_id", "nunique"),
                    unique_categories=("category", "nunique"),
                ).reset_index()

                X = feat[features_sel].copy()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Elbow method
                inertias = []
                K_range = range(2, 9)
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)

                km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                feat["cluster"] = km_final.fit_predict(X_scaled)
                feat["cluster_label"] = feat["cluster"].apply(lambda x: f"Segmento {x+1}")

            # ── Gráficos ──
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.subheader("📉 Método del Codo")
                fig_e, ax_e = plt.subplots(figsize=(6, 4))
                ax_e.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
                ax_e.axvline(x=n_clusters, color="red", linestyle="--", alpha=0.7, label=f"K={n_clusters} seleccionado")
                ax_e.set_xlabel("Número de clústeres (K)")
                ax_e.set_ylabel("Inercia (Distancia interna)")
                ax_e.set_title("Método del Codo — Elección de K")
                ax_e.legend()
                plt.tight_layout()
                st.pyplot(fig_e)
                plt.close()
                insight("El 'codo' ideal es donde la curva deja de bajar abruptamente. Ese es el K óptimo para tu negocio.")

            with col_g2:
                st.subheader("🗺️ Mapa de Segmentos")
                fig_s, ax_s = plt.subplots(figsize=(6, 4))
                for i, seg in enumerate(sorted(feat["cluster"].unique())):
                    mask = feat["cluster"] == seg
                    ax_s.scatter(feat.loc[mask, "total_spent"], feat.loc[mask, "order_count"],
                                 label=f"Segmento {seg+1}", alpha=0.7, s=40, color=PALETA[i % len(PALETA)])
                ax_s.set_xlabel("Gasto Total (S/)")
                ax_s.set_ylabel("Número de Órdenes")
                ax_s.set_title("Clientes por Gasto vs Frecuencia")
                ax_s.legend()
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close()
               # insight("Clientes arriba a la derecha = alta frecuencia y alto gasto = tus mejores clientes (VIP).")

            # ── Resumen por clúster ──
            st.subheader("📋 Perfiles de Segmentos")
            summary = feat.groupby("cluster_label")[features_sel].mean().round(2)
            summary.columns = [c.replace("_", " ").title() for c in summary.columns]

            # Asignar etiquetas automáticas
            if "total_spent" in features_sel and "order_count" in features_sel:
                raw_sum = feat.groupby("cluster")[["total_spent","order_count"]].mean()
                def etiquetar(row):
                    spent_pct = row["total_spent"] / raw_sum["total_spent"].max()
                    freq_pct  = row["order_count"] / raw_sum["order_count"].max()
                    if spent_pct > 0.6 and freq_pct > 0.6:
                        return "🏆 Cliente VIP"
                    elif spent_pct > 0.4 or freq_pct > 0.4:
                        return "🌱 Cliente Potencial"
                    else:
                        return "💤 Cliente Ocasional"
                etiquetas = raw_sum.apply(etiquetar, axis=1)
                summary["Perfil Automático"] = [etiquetas[i] for i in range(n_clusters)]

            st.dataframe(summary, use_container_width=True)

            # ── Recomendaciones por segmento ──
            #st.subheader("💡 Estrategias por Segmento")
            #estrategias = {
             #   "🏆 Cliente VIP": "**Programa de fidelización exclusivo.** Ofrece descuentos anticipados, envío gratis y acceso a productos nuevos antes del lanzamiento.",
             #   "🌱 Cliente Potencial": "**Campañas de activación.** Envía cupones de segunda compra, muestras gratis o bundle de productos complementarios.",
              #  "💤 Cliente Ocasional": "**Re-engagement.** Email de recuperación con oferta flash (48h), recordatorio de carrito y encuestas de satisfacción.",
            #}
            #for perfil, estrategia in estrategias.items():
             #   if "Perfil Automático" in summary.columns and perfil in summary["Perfil Automático"].values:
             #       st.markdown(f"**{perfil}:** {estrategia}")

            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                download_btn(feat, "segmentacion_clientes.xlsx", "Descargar Segmentación")
            with col_dl2:
                download_btn(summary.reset_index(), "resumen_segmentos.xlsx", "Descargar Resumen")

            st.session_state["clustering_done"] = feat

# ══════════════════════════════════════════════
# TAB 3 — NAIVE BAYES SENTIMENT
# ══════════════════════════════════════════════
with tab_sentiment:
    st.markdown('<div class="section-header">💬 Clasificación - Análisis de Sentimiento</div>', unsafe_allow_html=True)

    st.markdown("""
    **¿Qué hace?** Entrena un clasificador de texto que aprende a distinguir automáticamente entre reseñas 
    **Positivas**, **Negativas** y **Neutrales** basándose en las palabras usadas por los clientes.
    """)

    with st.expander("⚙️ Parámetros del Modelo", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("% datos de prueba:", 10, 40, 20,
                help="Porcentaje del dataset reservado para evaluar el modelo (no se usa para entrenamiento).")
        with col2:
            max_features = st.slider("Máx. palabras TF-IDF:", 100, 2000, 500,
                help="Las N palabras más relevantes que el modelo considerará para clasificar.")
        with col3:
            alpha_nb = st.slider("Suavizado Alpha (Laplace):", 0.1, 2.0, 1.0, step=0.1,
                help="Controla el suavizado del modelo. Valores más altos reducen el overfitting.")

    if st.button("▶️ Ejecutar Clasificación de Sentimiento", type="primary"):
        if df_r["sentiment"].nunique() < 2:
            warning_box("El dataset de reseñas necesita al menos 2 clases de sentimiento.")
        else:
            with st.spinner("Entrenando clasificador Naive Bayes..."):
                X_text = df_r["review_text"].fillna("")
                y = df_r["sentiment"]

                vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
                X_vec = vectorizer.fit_transform(X_text)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_vec, y, test_size=test_size/100, random_state=42, stratify=y
                )
                nb = MultinomialNB(alpha=alpha_nb)
                nb.fit(X_train, y_train)
                y_pred = nb.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.subheader(f"📊 Exactitud del Modelo: {acc*100:.1f}%")

                # Matriz de confusión
                labels = sorted(y.unique())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels, ax=ax_cm)
                ax_cm.set_xlabel("Predicho")
                ax_cm.set_ylabel("Real")
                ax_cm.set_title("Matriz de Confusión")
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

               # insight(f"La diagonal principal muestra predicciones correctas. Una exactitud de **{acc*100:.1f}%** significa que el modelo clasificó correctamente esa proporción de reseñas.")

            with col_g2:
                st.subheader("📈 Reporte de Clasificación")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).T.drop("accuracy", errors="ignore")
                report_df = report_df[["precision","recall","f1-score","support"]].round(3)
                report_df.columns = ["Precisión","Recall","F1-Score","Soporte"]
                st.dataframe(report_df, use_container_width=True)

                st.markdown("""
                **Guía de métricas:**
                - **Precisión**: De las que predijo como X, ¿cuántas eran realmente X?
                - **Recall**: De las que eran X, ¿cuántas identificó el modelo?
                - **F1-Score**: Promedio armónico entre Precisión y Recall (0=malo, 1=perfecto)
                """)

            # ── Top palabras por clase ──
            st.subheader("🔤 Palabras Más Importantes por Sentimiento")
            feature_names = vectorizer.get_feature_names_out()
            fig_w, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 4))
            if len(labels) == 1:
                axes = [axes]
            colors_map = {"Positivo": "#22c55e", "Neutral": "#eab308", "Negativo": "#ef4444"}
            for ax_w, (cls_idx, cls_name) in zip(axes, enumerate(nb.classes_)):
                top_idx = nb.feature_log_prob_[cls_idx].argsort()[-15:][::-1]
                top_words = [feature_names[i] for i in top_idx]
                top_scores = nb.feature_log_prob_[cls_idx][top_idx]
                ax_w.barh(top_words[::-1], top_scores[::-1],
                          color=colors_map.get(cls_name, "#4361ee"), alpha=0.85)
                ax_w.set_title(f"{cls_name}", fontweight="bold")
                ax_w.set_xlabel("Log-Probabilidad")
            plt.suptitle("Palabras más discriminativas por clase", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_w)
            plt.close()
            insight("Las palabras con mayor probabilidad son las que el modelo usa para decidir a qué clase pertenece una reseña nueva.")

            # ── Predictor en vivo ──
            st.markdown("---")
            st.subheader("🧪 Prueba el Modelo con una Nueva Reseña")
            nueva_reseña = st.text_area("Escribe una reseña:", placeholder="Ej: El producto llegó roto y tardó mucho...")
            if nueva_reseña.strip():
                vec_nueva = vectorizer.transform([nueva_reseña])
                pred = nb.predict(vec_nueva)[0]
                proba = nb.predict_proba(vec_nueva)[0]
                proba_dict = dict(zip(nb.classes_, proba))
                emoji_map = {"Positivo": "😊", "Neutral": "😐", "Negativo": "😠"}
                color_map = {"Positivo": "success", "Neutral": "warning", "Negativo": "error"}
                st.success(f"{emoji_map.get(pred, '🤔')} **Sentimiento predicho: {pred}**")
                st.write("**Probabilidades por clase:**")
                prob_df = pd.DataFrame({"Sentimiento": list(proba_dict.keys()), "Probabilidad": list(proba_dict.values())})
                fig_prob, ax_prob = plt.subplots(figsize=(5, 2.5))
                colors_p = [colors_map.get(s, "#999") for s in prob_df["Sentimiento"]]
                ax_prob.barh(prob_df["Sentimiento"], prob_df["Probabilidad"], color=colors_p)
                ax_prob.set_xlim(0, 1)
                ax_prob.set_xlabel("Probabilidad")
                ax_prob.set_title("Confianza del Modelo")
                plt.tight_layout()
                st.pyplot(fig_prob)
                plt.close()

            # ── Descarga ──
            df_r_pred = df_r.copy()
            df_r_pred["predicted_sentiment"] = nb.predict(vectorizer.transform(df_r_pred["review_text"].fillna("")))
            st.markdown("---")
            download_btn(df_r_pred, "reseñas_clasificadas.xlsx", "Descargar Reseñas Clasificadas")

# ══════════════════════════════════════════════
# TAB 4 — APRIORI ASSOCIATION RULES
# ══════════════════════════════════════════════
with tab_apriori:
    st.markdown('<div class="section-header">🔗 Reglas de Asociación — Algoritmo Apriori</div>', unsafe_allow_html=True)

    st.markdown("""
    **¿Qué hace?** Descubre qué productos se compran juntos con frecuencia.
    Es la base del sistema de recomendación "Los clientes que compraron X también compraron Y".
    """)

    with st.expander("⚙️ Parámetros del Modelo", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider("Soporte Mínimo:", 0.0001, 0.05, 0.01, step=0.0001, format="%.4f",
                help="Frecuencia mínima con la que el conjunto de productos debe aparecer en las órdenes.")
        with col2:
            min_confidence = st.slider("Confianza Mínima:", 0.1, 1.0, 0.3, step=0.05,
                help="Probabilidad de que si se compra A, también se compre B.")
        with col3:
            min_lift = st.slider("Lift Mínimo:", 1.0, 5.0, 2.0, step=0.1,
                help="Lift > 1 significa que los productos se compran juntos más de lo esperado por azar.")

        top_n = st.slider("Top N reglas a mostrar:", 5, 50, 20)

    if st.button("▶️ Ejecutar Análisis de Asociación", type="primary"):
        with st.spinner("Ejecutando algoritmo Apriori... (puede tardar un momento)"):
            try:
                basket = (
                    df_t.groupby(["order_id", "product_name"])["quantity"]
                    .sum().unstack().fillna(0)
                    .applymap(lambda x: 1 if x > 0 else 0)
                )

                frequent_items = apriori(basket, min_support=min_support, use_colnames=True)

                if frequent_items.empty:
                    warning_box(f"No se encontraron conjuntos frecuentes con soporte={min_support:.4f}. Reduce el soporte mínimo.")
                else:
                    rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift)
                    rules = rules[rules["confidence"] >= min_confidence]
                    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
                    rules = rules.sort_values("lift", ascending=False).head(top_n)

                    if rules.empty:
                        warning_box("No hay reglas con esos parámetros. Prueba reducir confianza o lift mínimo.")
                    else:
                        success_box(f"Se encontraron {len(frequent_items)} conjuntos frecuentes y {len(rules)} reglas de asociación.")

                        col_g1, col_g2 = st.columns(2)

                        with col_g1:
                            st.subheader("📊 Lift vs Confianza")
                            fig_lr, ax_lr = plt.subplots(figsize=(7, 5))
                            scatter = ax_lr.scatter(
                                rules["confidence"], rules["lift"],
                                c=rules["support"], cmap="viridis",
                                s=rules["support"]*5000, alpha=0.7, edgecolors="white", linewidth=0.5
                            )
                            plt.colorbar(scatter, ax=ax_lr, label="Soporte")
                            ax_lr.set_xlabel("Confianza →")
                            ax_lr.set_ylabel("Lift →")
                            ax_lr.set_title("Mapa de Calidad de Reglas")
                            ax_lr.axvline(0.5, color="red", alpha=0.4, linestyle="--", label="Conf. 0.5")
                            ax_lr.axhline(1.5, color="blue", alpha=0.4, linestyle="--", label="Lift 1.5")
                            ax_lr.legend(fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig_lr)
                            plt.close()
                            insight("Las reglas en la esquina superior derecha son las más valiosas: alta confianza Y alto lift.")

                        with col_g2:
                            st.subheader("🔝 Top Reglas por Lift")
                            top10 = rules.head(10)
                            fig_top, ax_top = plt.subplots(figsize=(7, 5))
                            labels_r = [f"{a[:20]}→{c[:15]}" for a,c in zip(top10["antecedents"], top10["consequents"])]
                            ax_top.barh(labels_r[::-1], top10["lift"].values[::-1], color=PALETA[0], alpha=0.85)
                            ax_top.axvline(1, color="red", linestyle="--", alpha=0.6, label="Lift=1 (azar)")
                            ax_top.set_xlabel("Lift")
                            ax_top.set_title("Top 10 Reglas por Lift")
                            ax_top.legend(fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig_top)
                            plt.close()

                        # ── Tabla de reglas ──
                        st.subheader("📋 Tabla de Reglas de Asociación")
                        display_rules = rules[["antecedents","consequents","support","confidence","lift"]].copy()
                        display_rules.columns = ["Si compra (A)","También compra (B)","Soporte","Confianza","Lift"]
                        display_rules = display_rules.style.format({
                            "Soporte": "{:.4f}", "Confianza": "{:.3f}", "Lift": "{:.3f}"
                        }).background_gradient(subset=["Lift"], cmap="YlOrRd")
                        st.dataframe(display_rules, use_container_width=True)

                        st.markdown("""
                        **📖 Guía de interpretación:**
                        | Métrica | ¿Qué significa? | Valor ideal |
                        |---------|----------------|-------------|
                        | **Soporte** | % de órdenes que contienen AMBOS productos | > 0.01 |
                        | **Confianza** | Si compra A, probabilidad de comprar B | > 0.5 |
                        | **Lift** | ¿Cuánto más probable que comprarlo por azar? | > 1.5 |
                        """)

                        insight("Un Lift de 2.5 significa que esos productos se compran juntos 2.5 veces más de lo que ocurriría por azar. Úsalo para diseñar combos, ofertas 2x1 o posición de productos en la tienda.")

                        st.markdown("---")
                        download_btn(rules.reset_index(drop=True), "reglas_asociacion.xlsx", "Descargar Reglas de Asociación")

            except Exception as e:
                st.error(f"Error al ejecutar Apriori: {e}")
                warning_box("Intenta reducir el soporte mínimo o verificar que el dataset tenga suficientes órdenes con múltiples productos.")

# ══════════════════════════════════════════════
# TAB 5 — ANÁLISIS GEOGRÁFICO
# ══════════════════════════════════════════════
with tab_geo:
    st.markdown('<div class="section-header">🗺️ Análisis Geográfico de Ventas</div>', unsafe_allow_html=True)

    st.markdown("""
    **¿Qué hace?** Visualiza cómo se distribuyen las ventas, clientes y ticket promedio por zona geográfica.
    Ideal para planificar expansión, logística y campañas locales.
    """)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("💰 Ventas Totales por Distrito")
        ventas_geo = df_t.groupby("place")["total_amount"].sum().sort_values(ascending=False).head(15)
        fig_geo, ax_geo = plt.subplots(figsize=(7, 5))
        colors_geo = [PALETA[i % len(PALETA)] for i in range(len(ventas_geo))]
        bars = ax_geo.bar(range(len(ventas_geo)), ventas_geo.values, color=colors_geo)
        ax_geo.set_xticks(range(len(ventas_geo)))
        ax_geo.set_xticklabels(ventas_geo.index, rotation=45, ha="right", fontsize=8)
        ax_geo.set_ylabel("Ventas (S/)")
        ax_geo.set_title("Top 15 Distritos por Ventas")
        plt.tight_layout()
        st.pyplot(fig_geo)
        plt.close()
        #insight(f"El distrito líder en ventas es **{ventas_geo.index[0]}**. #Considera abrir un punto de atención o mejorar la cobertura de delivery en #las zonas top.")

    with col_g2:
        st.subheader("📦 Órdenes y Ticket Promedio por Zona")
        geo_full = df_t.groupby("place").agg(
            total_ventas=("total_amount", "sum"),
            total_ordenes=("order_id", "nunique"),
            ticket_prom=("total_amount", "mean"),
            clientes_unicos=("customer_id", "nunique"),
        ).sort_values("total_ventas", ascending=False).head(15).reset_index()
        geo_full.columns = ["Distrito","Ventas (S/)","Órdenes","Ticket Prom","Clientes"]
        geo_full["Ventas (S/)"] = geo_full["Ventas (S/)"].round(0)
        geo_full["Ticket Prom"] = geo_full["Ticket Prom"].round(1)
        st.dataframe(
            geo_full.style.background_gradient(subset=["Ventas (S/)"], cmap="Blues")
                          .background_gradient(subset=["Ticket Prom"], cmap="Greens"),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🏷️ Categoría más vendida por Distrito")
    top_cat_geo = (
        df_t.groupby(["place","category"])["total_amount"].sum()
        .reset_index()
        .sort_values("total_amount", ascending=False)
        .groupby("place")
        .first()
        .reset_index()
        .rename(columns={"category":"Categoría líder","total_amount":"Ventas (S/)"})
        .sort_values("Ventas (S/)", ascending=False)
        .head(15)
    )
    st.dataframe(top_cat_geo, use_container_width=True)
   # insight("Conocer la categoría líder por distrito te permite adaptar el #mix de productos según la demanda local.")

    download_btn(geo_full, "analisis_geografico.xlsx", "Descargar Análisis Geográfico")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-top:60px; padding:20px;
            background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-radius:12px; color:#a0aec0; font-size:13px;'>
    🧠 <b style='color:white'>Data Mining Simulator</b> — Aplicando IA al análisis de e-commerce peruano<br>
    Técnicas: K-Means · Naive Bayes · Apriori | Desarrollado por Wilton Torvisco para uso educativo
</div>
""", unsafe_allow_html=True)
