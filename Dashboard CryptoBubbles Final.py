# https://chatgpt.com/c/68c5c464-aa24-832c-813a-017b6f105565
# testes_rápidos\Scripts\activate // Em ambiente virtual C:\Users\Usuário\Documents\_Projetos\APIs\
# streamlit run 'Dashboard CryptoBubbles49 Grok.py'
# 	You can now view your Streamlit app in your browser.
# 	Local URL: http://localhost:8501
# 	Network URL: http://192.168.1.3:8501

# Funcionou tudo dinamicamente

from io import BytesIO

import pandas as pd
import requests
import streamlit as st

URL = "https://cryptobubbles.net/backend/data/bubbles1000.usd.json"

# ----------------------------
# Funções auxiliares
# ----------------------------
@st.cache_data(ttl=60)  # Cache por 1 minuto
def obter_dados(url=URL):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def normalizar_json(dados):
    return pd.json_normalize(dados, sep=".")

def converter_para_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Dados")
    return output.getvalue()

def human_format(num):
    if num is None or pd.isna(num):
        return ""
    num = float(num)
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:,.2f}"

def color_performance(val):
    if pd.isna(val):
        return ""
    val_float = float(val)
    if val_float > 0:
        return "background-color: #056F05; color: white"
    elif val_float < 0:
        return "background-color: #A80606; color: white"
    else:
        return ""

# ----------------------------
# Configuração Streamlit
# ----------------------------
st.set_page_config(page_title="Crypto Dashboard BI", layout="wide")
# Tema dark agora configurado via .streamlit/config.toml - CSS manual comentado como backup
# st.markdown(
#     """
#     <style>
#     .main { background-color: #0e1117 !important; color: white !important; }
#     h1, h2, h3, h4 { color: #f5f5f5 !important; }
#     .stDataFrame table { background-color: #1e2229 !important; color: white !important; }
#     .stApp { background-color: #0e1117 !important; }
#     .stButton>button { background-color: #1e2229 !important; color: white !important; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title("💼 Dashboard Executivo - Criptomoedas")
st.markdown("📊 Fonte: [CryptoBubbles API](https://cryptobubbles.net/)")

# ----------------------------
# Carregar dados
# ----------------------------
json_data = obter_dados()
df = normalizar_json(json_data)
df_total = df  # Armazena os dados completos antes de aplicar filtros
if "marketcap" in df.columns:
    df = df[df["marketcap"] >= 100_000_000]

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Filtros e Chaves")
default_keys = [
    "name",
    "symbol",
    "price",
    "marketcap",
    "volume",
    "dominance",
    "performance.min15",
    "performance.hour",
    "performance.hour4",
    "performance.day",
    "performance.week",
    "performance.month",
    "performance.month3",
    "performance.year",
]

# Inicialização das session_state
if "selected_keys" not in st.session_state:
    st.session_state.selected_keys = default_keys.copy()
if "slider_values" not in st.session_state:
    st.session_state.slider_values = {}
if "multiselect_key_counter" not in st.session_state:
    st.session_state.multiselect_key_counter = 0

# Pré-popula os valores do slider com base nos dados do DataFrame
if not st.session_state.slider_values:
    for c in st.session_state.selected_keys:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            min_val = float(df[c].min())
            max_val = float(df[c].max())
            st.session_state.slider_values[c] = (min_val, max_val)

# Callback para restaurar chaves default
def restore_keys_callback():
    st.session_state.selected_keys = default_keys.copy()
    st.session_state.multiselect_key_counter += 1
    st.session_state.slider_values = {}
    st.session_state.update_inputs = True  # Sinaliza a atualização

# Callback para restaurar filtros default
def restore_filters_callback():
    for c in st.session_state.selected_keys:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            min_val = float(df[c].min())
            max_val = float(df[c].max())
            st.session_state.slider_values[c] = (min_val, max_val)
    st.session_state.update_inputs = True  # Sinaliza a atualização

# Botões usam on_click com callbacks
st.sidebar.button(
    "🔄 Restaurar Chaves Default",
    on_click=restore_keys_callback,
    key="restore_keys_btn",
)
st.sidebar.button(
    "🔄 Restaurar Filtros Default",
    on_click=restore_filters_callback,
    key="restore_filters_btn",
)

# Multiselect com key dinâmico
multiselect_key = f"multiselect_{st.session_state.multiselect_key_counter}"
chaves_selecionadas = st.sidebar.multiselect(
    "📋 Escolha chaves para consulta",
    options=[*df.columns],
    default=st.session_state.selected_keys,
    key=multiselect_key,
)
st.session_state.selected_keys = chaves_selecionadas

# ----------------------------
# Sliders + input manual para filtros numéricos
# ----------------------------
for c in chaves_selecionadas:
    if pd.api.types.is_numeric_dtype(df[c]):
        min_val_df = float(df[c].min())
        max_val_df = float(df[c].max())

        # Obter os valores atuais do slider/inputs
        current_min, current_max = st.session_state.slider_values.get(
            c, (min_val_df, max_val_df)
        )

        # Callback para atualizar os valores de min e max quando os inputs mudam
        def update_slider_from_inputs(key):
            min_input = st.session_state.get(f"{key}_min")
            max_input = st.session_state.get(f"{key}_max")

            # Garante que os valores de min e max são válidos
            if (
                min_input is not None
                and max_input is not None
                and min_input <= max_input
            ):
                st.session_state.slider_values[key] = (min_input, max_input)
            elif min_input is not None and max_input is None:
                st.session_state.slider_values[key] = (min_input, max_val_df)
            elif max_input is not None and min_input is None:
                st.session_state.slider_values[key] = (min_val_df, max_input)

        # Sliders e number_inputs para cada coluna
        selected_range = st.sidebar.slider(
            f"{c} ({human_format(min_val_df)} - {human_format(max_val_df)})",
            min_value=min_val_df,
            max_value=max_val_df,
            value=(current_min, current_max),
            key=f"{c}_slider",
            on_change=lambda c=c: st.session_state.slider_values.update(
                {c: st.session_state[f"{c}_slider"]}
            ),
        )

        st.sidebar.number_input(
            f"Min {c}",
            min_value=min_val_df,
            max_value=max_val_df,
            value=selected_range[0],
            key=f"{c}_min",
            on_change=update_slider_from_inputs,
            args=(c,),
        )
        st.sidebar.number_input(
            f"Max {c}",
            min_value=min_val_df,
            max_value=max_val_df,
            value=selected_range[1],
            key=f"{c}_max",
            on_change=update_slider_from_inputs,
            args=(c,),
        )

        st.sidebar.markdown("---")

# Filtrar o DataFrame com base nos valores em st.session_state.slider_values
df_filtrado = df[chaves_selecionadas].copy()
for c, (min_val, max_val) in st.session_state.slider_values.items():
    if c in df_filtrado.columns:
        df_filtrado = df_filtrado[
            (df_filtrado[c] >= min_val) & (df_filtrado[c] <= max_val)
        ]

# ----------------------------
# Performance float
# ----------------------------
perf_keys = [
    "performance.min1",
    "performance.min5",
    "performance.min15",
    "performance.hour",
    "performance.hour4",
    "performance.day",
    "performance.week",
    "performance.month",
    "performance.month3",
    "performance.year",
]
for k in perf_keys:
    if k in df_filtrado.columns:
        df_filtrado[k + "_float"] = pd.to_numeric(df_filtrado[k], errors="coerce")

# ----------------------------
# Resultado da Consulta Multi-Nível
# ----------------------------
st.markdown("## 🔎 Resultado da Consulta Multi-Nível")
# Criar df_display diretamente de df_filtrado, preservando todas as linhas
df_display = df_filtrado.copy()

# Garantir que colunas numéricas sejam float para sorting, tratando NaN
for col in ["marketcap", "volume", "price"]:
    if col in df_display.columns:
        df_display[col] = pd.to_numeric(df_display[col], errors="coerce", downcast="float").fillna(0)

# Criar colunas formatadas para exibição legível
for col in ["marketcap", "volume", "price"]:
    if col in df_display.columns:
        df_display[f"{col}_formatted"] = df_display[col].apply(human_format)

# Tratar todas as colunas de performance, preenchendo NaN com "" para exibição
for k in perf_keys:
    if k in df_display.columns:
        df_display[k] = df_display[k].fillna("")

# Renomear colunas performance.* sem alterar o DataFrame subjacente
renomear = {
    "performance.min1": "perf.min1",
    "performance.min5": "perf.min5",
    "performance.min15": "perf.min15",
    "performance.hour": "perf.hour",
    "performance.hour4": "perf.hour4",
    "performance.day": "perf.day",
    "performance.week": "perf.week",
    "performance.month": "perf.month",
    "performance.month3": "perf.month3",
    "performance.year": "perf.year",
}
df_display = df_display.rename(columns=renomear)

# Eliminar colunas _float após uso
for k in perf_keys:
    float_col = k + "_float"
    if float_col in df_display.columns:
        df_display.drop(columns=[float_col], inplace=True)

# Configuração de colunas
column_config = {}
perf_novas = [new_col for old_col, new_col in renomear.items() if old_col in df_filtrado.columns]

# Para performance
for col in perf_novas:
    if col in df_display.columns:
        column_config[col] = st.column_config.TextColumn(
            default="",  # Permite exibir "" para NaN
        )

# Para marketcap, volume, price
for col in ["marketcap", "volume", "price"]:
    if col in df_display.columns:
        column_config[col] = st.column_config.NumberColumn(
            label=col.replace("_", " ").title(), disabled=True
        )
        formatted_col = f"{col}_formatted"
        column_config[formatted_col] = st.column_config.TextColumn(
            label=f"{col.replace('_', ' ').title()} (Formatted)", default=""
        )

display_columns = [
    col
    for col in df_display.columns
    if col.endswith("_formatted")
    or col in ["price", "marketcap", "volume", "name", "symbol", "dominance"]
    or col in perf_novas
]

if not df_display.empty:
    valid_perf_cols = [col for col in perf_novas if col in display_columns]
    st.dataframe(
        df_display[display_columns].style.map(
            color_performance, subset=valid_perf_cols if valid_perf_cols else None
        ),
        height=500,
        column_config=column_config if column_config else None,
        hide_index=False,
    )
else:
    st.warning("⚠️ Nenhuma chave válida selecionada.")

# ----------------------------
# Alertas Top 3 Performance
# ----------------------------
st.markdown("## 🚨 Alertas Top 3 Performance")
intervalos = {
    "performance.hour": "Hora",
    "performance.day": "Dia",
    "performance.week": "Semana",
    "performance.month": "Mês",
    "performance.month3": "3 Meses",
}
colunas_essenciais = ["name", "symbol", "price"]
colunas_disponiveis = [col for col in colunas_essenciais if col in df_filtrado.columns]

for key, label in intervalos.items():
    float_key = key + "_float"
    if float_key in df_filtrado.columns and not df_filtrado.empty:
        cols_para_selecao = colunas_disponiveis + [float_key]
        top_altas = df_filtrado.nlargest(3, float_key)[cols_para_selecao]
        top_baixas = df_filtrado.nsmallest(3, float_key)[cols_para_selecao]
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"### 🟢 Maiores Altas - {label}")
            for idx, (_, r) in enumerate(top_altas.iterrows(), 1):
                nome_str = (
                    f"{r.get('name', 'N/A')} " if "name" in colunas_disponiveis else ""
                )
                symbol_str = (
                    f"({r.get('symbol', 'N/A')})"
                    if "symbol" in colunas_disponiveis
                    else ""
                )
                preco_str = (
                    f" - ${human_format(r.get('price'))}"
                    if "price" in colunas_disponiveis
                    else ""
                )
                linha_str = (
                    f" [# {df_display.index.get_loc(r.name) + 1}]"
                    if not df_display.empty and r.name in df_display.index
                    else ""
                )
                st.markdown(
                    f"- {nome_str}{symbol_str}{preco_str}: {r[float_key]:.2f}%{linha_str}"
                )
        with colB:
            st.markdown(f"### 🔴 Maiores Baixas - {label}")
            for idx, (_, r) in enumerate(top_baixas.iterrows(), 1):
                nome_str = (
                    f"{r.get('name', 'N/A')} " if "name" in colunas_disponiveis else ""
                )
                symbol_str = (
                    f"({r.get('symbol', 'N/A')})"
                    if "symbol" in colunas_disponiveis
                    else ""
                )
                preco_str = (
                    f" - ${human_format(r.get('price'))}"
                    if "price" in colunas_disponiveis
                    else ""
                )
                linha_str = (
                    f" [# {df_display.index.get_loc(r.name) + 1}]"
                    if not df_display.empty and r.name in df_display.index
                    else ""
                )
                st.markdown(
                    f"- {nome_str}{symbol_str}{preco_str}: {r[float_key]:.2f}%{linha_str}"
                )
    else:
        st.warning(f"⚠️ Dados insuficientes para alertas de {label}.")

# ----------------------------
# Exportação
# ----------------------------
st.markdown("## 💾 Exportação de Dados")
if not df_display.empty:
    df_export = df_display.copy()
    for col in ["marketcap", "volume", "price"]:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(human_format)
    dados_excel = converter_para_excel(df_export[display_columns])
    st.download_button("📥 Baixar dados filtrados", dados_excel, "dados_filtrados.xlsx")

# ----------------------------
# Estatísticas de Desempenho
# ----------------------------
st.markdown("## 📊 Estatísticas de Desempenho")
if not df.empty:
    mean_values = df[perf_keys].mean(numeric_only=True).round(2)
    median_values = df[perf_keys].median(numeric_only=True).round(2)
    std_values = df[perf_keys].std(numeric_only=True).round(2)
    min_values = df[perf_keys].min(numeric_only=True).round(2)
    max_values = df[perf_keys].max(numeric_only=True).round(2)

    performance_labels = [
        "Min1",
        "Min5",
        "Min15",
        "Hour",
        "Hour4",
        "Day",
        "Week",
        "Month",
        "Month3",
        "Year",
    ]
    stats_df = pd.DataFrame(
        {
            "Média (%)": [f"{val:.2f}%" for val in mean_values],
            "Mediana (%)": [f"{val:.2f}%" for val in median_values],
            "Desvio Padrão (%)": [f"{val:.2f}%" for val in std_values],
            "Mínimo (%)": [f"{val:.2f}%" for val in min_values],
            "Máximo (%)": [f"{val:.2f}%" for val in max_values],
        },
        index=performance_labels,
    )

    st.dataframe(
        stats_df.style.set_properties(
            **{
                "background-color": "#1e2229",
                "color": "white",
                "border": "1px solid #333",
                "text-align": "center",
                "font-size": "14px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#0e1117"),
                        ("color", "#f5f5f5"),
                        ("border", "1px solid #333"),
                        ("text-align", "center"),
                        ("font-size", "14px"),
                    ],
                }
            ]
        )
        .map(lambda x: "", subset=None),
        height=385,
    )
else:
    st.warning("⚠️ Não há dados disponíveis para calcular estatísticas.")

# ----------------------------
# Estatísticas de Desempenho Filtrada
# ----------------------------
st.markdown("## 📊 Estatísticas de Desempenho Filtrada")
if not df_total.empty:
    # Campo de entrada para número de criptomoedas
    num_cryptos = st.number_input(
        "Número de criptomoedas a considerar (1-1000)",
        min_value=1,
        max_value=1000,
        value=100,
        key="num_cryptos_input",
    )

    # Selecionar as top N criptomoedas por marketcap do df_total
    df_top_cryptos = (
        df_total.nlargest(int(num_cryptos), "marketcap")
        if "marketcap" in df_total.columns
        else df_total
    )

    # Usar apenas as colunas de performance disponíveis em df_total
    perf_keys_available = [k for k in perf_keys if k in df_total.columns]

    # Calcular estatísticas com colunas disponíveis
    mean_values = df_top_cryptos[perf_keys_available].mean(numeric_only=True).round(2)
    median_values = df_top_cryptos[perf_keys_available].median(numeric_only=True).round(2)
    std_values = df_top_cryptos[perf_keys_available].std(numeric_only=True).round(2)
    min_values = df_top_cryptos[perf_keys_available].min(numeric_only=True).round(2)
    max_values = df_top_cryptos[perf_keys_available].max(numeric_only=True).round(2)

    # Ajustar performance_labels para corresponder às colunas disponíveis
    performance_labels = [
        label
        for label, key in zip(
            [
                "Min1",
                "Min5",
                "Min15",
                "Hour",
                "Hour4",
                "Day",
                "Week",
                "Month",
                "Month3",
                "Year",
            ],
            perf_keys,
        )
        if key in perf_keys_available
    ]

    stats_df_filtered = pd.DataFrame(
        {
            "Média (%)": [f"{val:.2f}%" for val in mean_values],
            "Mediana (%)": [f"{val:.2f}%" for val in median_values],
            "Desvio Padrão (%)": [f"{val:.2f}%" for val in std_values],
            "Mínimo (%)": [f"{val:.2f}%" for val in min_values],
            "Máximo (%)": [f"{val:.2f}%" for val in max_values],
        },
        index=performance_labels,
    )

    st.dataframe(
        stats_df_filtered.style.set_properties(
            **{
                "background-color": "#1e2229",
                "color": "white",
                "border": "1px solid #333",
                "text-align": "center",
                "font-size": "14px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#0e1117"),
                        ("color", "#f5f5f5"),
                        ("border", "1px solid #333"),
                        ("text-align", "center"),
                        ("font-size", "14px"),
                    ],
                }
            ]
        )
        .map(lambda x: "", subset=None),
        height=385,
    )
else:
    st.warning("⚠️ Não há dados disponíveis para calcular estatísticas.")
