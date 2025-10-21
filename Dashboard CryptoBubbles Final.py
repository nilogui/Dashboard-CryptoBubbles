from io import BytesIO
import pandas as pd
import requests
import streamlit as st
import time
import numpy as np

URL = "https://cryptobubbles.net/backend/data/bubbles1000.usd.json"

# ----------------------------
# Funções auxiliares
# ----------------------------


@st.cache_data(ttl=120)  # Caching: Dados são estáticos por 120 segundos
def obter_dados(url=URL):
    """Obtém os dados do Cryptobubbles."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def otimizar_dtypes(df):
    """Reduz o uso de memória do DataFrame alterando os dtypes numéricos."""
    for col in df.columns:
        # Apenas para colunas numéricas (inteiro e float)
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            # Tipos de inteiro
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)

            # Tipos de ponto flutuante
            elif pd.api.types.is_float_dtype(df[col]):
                # Cuidado: Reduzir para float16 pode perder precisão (price usa até 8 casas)
                # Mantemos float32 (ou float64) para 'price', mas tentamos float32 para as outras.

                # Exemplo para Performance (com 2 casas decimais)
                if "performance" in col or "dominance" in col or "volume" in col:
                    # Garante que não há NaNs antes de checar os limites para float
                    df[col] = df[col].fillna(
                        0
                    )  # Trata NaNs temporariamente para downcast

                    if (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)

                    # Se 'price' precisa de alta precisão (8 casas), mantenha o original ou float64
                    # O downcasting de float é mais complexo devido à precisão.

    return df


@st.cache_data  # Caching: Normaliza o JSON APENAS se os dados de entrada mudarem
def normalizar_json(dados):
    """Normaliza o JSON e adiciona a coluna de rank."""
    df = pd.json_normalize(dados, sep=".")
    df["rank"] = df["marketcap"].rank(method="min", ascending=False).astype(int)
    return otimizar_dtypes(df)  # return df


def converter_para_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Dados")
    return output.getvalue()


def color_performance(val):
    if pd.isna(val) or val is None:
        return ""
    try:
        val_float = float(val)
        if val_float > 0:
            return "background-color: #228822; color: white"
        elif val_float < 0:
            return "background-color: #AA3333; color: white"
        else:
            return ""
    except (ValueError, TypeError):
        return ""


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


# ----------------------------
# Configuração Streamlit
# ----------------------------
st.set_page_config(page_title="Crypto Dashboard BI", layout="wide")
st.title("💼 Dashboard Executivo - Criptomoedas")
st.markdown("📊 Fonte: [CryptoBubbles API](https://cryptobubbles.net/)")

# ----------------------------
# Carregar dados e gerar links
# ----------------------------
json_data = obter_dados()
df = normalizar_json(json_data)
df_total = df.copy()

# 1. Coluna do CoinMarketCap
df_total["links"] = df_total["slug"].apply(
    lambda x: f"https://coinmarketcap.com/currencies/{x}" if x else ""
)

# 2. Coluna do CoinGecko (usando cg_id)
df_total["coingecko_links"] = df_total["cg_id"].apply(
    lambda x: f"https://www.coingecko.com/en/coins/{x}" if x else ""
)

# 3. Coluna do TradingView - Link Rápido (Overview)
df_total["tradingview_links"] = df_total["symbol"].apply(
    lambda x: f"https://www.tradingview.com/symbols/{x}USDT" if x else ""
)

# 4. Coluna do TradingView - Link Gráfico (Chart)
df_total["tradingview_links_chart"] = df_total["symbol"].apply(
    lambda x: f"https://www.tradingview.com/chart/?symbol={x}USDT" if x else ""
)


# ----------------------------
# Sidebar (Ordem Ajustada)
# ----------------------------
st.sidebar.header("⚙️ Configurações e Filtros")

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
    "links",
    "coingecko_links",
    "tradingview_links",
    "tradingview_links_chart",
]
perf_keys = [k for k in default_keys if "performance" in k]

if "selected_keys" not in st.session_state:
    st.session_state.selected_keys = default_keys.copy()
if "slider_values" not in st.session_state:
    st.session_state.slider_values = {}
if "multiselect_key_counter" not in st.session_state:
    st.session_state.multiselect_key_counter = 0
if "multiselect_temp_key" not in st.session_state:
    st.session_state.multiselect_temp_key = default_keys.copy()


marketcap_options = {
    "0": 0.0,
    "100M": 100_000_000.0,
    "500M": 500_000_000.0,
    "1B": 1_000_000_000.0,
    "5B": 5_000_000_000.0,
    "10B": 10_000_000_000.0,
}

quick_select_options = {
    "Todas": default_keys,
    "Performances": ["name", "price"]
    + [k for k in default_keys if "performance" in k]
    + ["links", "coingecko_links", "tradingview_links", "tradingview_links_chart"],
    "Volume/Dominance": [
        "name",
        "price",
        "volume",
        "dominance",
        "performance.day",
        "performance.week",
        "performance.month",
        "performance.month3",
        "performance.year",
        "links",
        "coingecko_links",
        "tradingview_links",
        "tradingview_links_chart",
    ],
}


# CRÍTICO: Função de callback para garantir que o valor do multiselect seja copiado na hora
def multiselect_callback():
    # Copia o valor do widget (que está na chave multiselect_X) para o estado persistente
    st.session_state.selected_keys = st.session_state[
        f"multiselect_{st.session_state.multiselect_key_counter}"
    ]


# Função para forçar a atualização da seleção e re-renderizar o multiselect
def update_selected_keys():
    selected_option = st.session_state.quick_select_option
    st.session_state.selected_keys = quick_select_options[selected_option]
    # Força a re-renderização mudando a chave do multiselect
    st.session_state.multiselect_key_counter += 1


def update_marketcap_min():
    """Atualiza o filtro de marketcap com base na seleção rápida."""
    marketcap_key = "marketcap"
    selected_min_cap = marketcap_options[st.session_state.marketcap_quick_select]

    # 1. Apenas atualiza o slider_values (o que realmente filtra)
    if marketcap_key not in st.session_state.slider_values:
        max_val_marketcap = round(float(df_total["marketcap"].max()) * 1.1, 2)
        st.session_state.slider_values[marketcap_key] = (
            selected_min_cap,
            max_val_marketcap,
        )
    else:
        current_max = st.session_state.slider_values[marketcap_key][1]
        st.session_state.slider_values[marketcap_key] = (selected_min_cap, current_max)

    # 2. Atualiza a key do number_input para refletir a mudança
    st.session_state[f"{marketcap_key}_min"] = selected_min_cap


def restore_keys_callback():
    st.session_state.selected_keys = default_keys.copy()
    # Força a re-renderização mudando a chave do multiselect
    st.session_state.multiselect_key_counter += 1


def restore_filters_callback():
    """Restaura todos os filtros numéricos para o valor padrão."""
    for c in df_total.columns:
        if c in default_keys and pd.api.types.is_numeric_dtype(df_total[c]):
            min_val = float(df_total[c].min())
            max_val = float(df_total[c].max())

            # --- Regra de Limite do Widget (com margem de 10%/20% e arredondamento) ---
            if c == "marketcap":
                # Limite do widget: limite absoluto do DF
                min_val_limit = round(float(df_total[c].min()), 2)
                max_val_limit = round(max_val * 1.1, 2)

                # Valor inicial do state (usa 100M)
                min_val_state = marketcap_options["100M"]
                max_val_state = max_val_limit

            elif c != "price" and c != "dominance":
                # Limite do Widget (Min_value e Max_value)
                min_val_limit = round(
                    min_val * 0.9 if min_val > 0 else min_val * 1.1, 2
                )
                max_val_limit = round(
                    max_val * 1.1 if max_val > 0 else max_val * 0.9, 2
                )

                # Valor inicial do state (usa o limite calculado)
                min_val_state = min_val_limit
                max_val_state = max_val_limit

            elif c == "price":
                min_val_limit = 0.0
                # CORREÇÃO CRÍTICA: Aumenta a margem para 20% no price para evitar erros de float
                max_val_limit = round(df_total["price"].max() * 1.2, 2)

                min_val_state = 0.0
                max_val_state = max_val_limit

            elif c == "dominance":
                min_val_limit = 0.0
                max_val_limit = round(max_val * 1.1, 2)

                min_val_state = 0.0
                max_val_state = max_val_limit

            # --- Atribuição ao Session State ---
            # Garante que o valor no state não é menor/maior que o limite do widget
            st.session_state.slider_values[c] = (min_val_state, max_val_state)

            st.session_state[f"{c}_min"] = min_val_state
            st.session_state[f"{c}_max"] = max_val_state


# --- 1. Marketcap Mínimo (Seleção Rápida) ---
# Inicialização dos valores do Market Cap (Market Cap Mínimo é 100M por padrão)
if "marketcap" in df_total.columns:
    max_val_marketcap = round(float(df_total["marketcap"].max()) * 1.1, 2)
    min_val_marketcap_default = marketcap_options["100M"]

    # Inicializa slider_values
    if "marketcap" not in st.session_state.slider_values:
        st.session_state.slider_values["marketcap"] = (
            min_val_marketcap_default,
            max_val_marketcap,
        )

    # Inicialização das keys dos number_inputs
    if "marketcap_min" not in st.session_state:
        # Usa o valor do slider (100M)
        st.session_state["marketcap_min"] = st.session_state.slider_values["marketcap"][
            0
        ]
    if "marketcap_max" not in st.session_state:
        st.session_state["marketcap_max"] = st.session_state.slider_values["marketcap"][
            1
        ]


if "marketcap" in st.session_state.selected_keys or "marketcap" in df_total.columns:
    marketcap_quick_select = st.sidebar.selectbox(
        "Marketcap Mínimo (Seleção Rápida)",
        options=list(marketcap_options.keys()),
        index=list(marketcap_options.keys()).index("100M"),
        key="marketcap_quick_select",
        on_change=update_marketcap_min,
        help="Define o Market Cap mínimo que uma criptomoeda deve ter. O valor é refletido no slider de Marketcap.",
    )
    st.sidebar.markdown("---")

# --- 2. Número de criptomoedas na tabela (Input Numérico) ---
num_cryptos_table = st.sidebar.number_input(
    "Número de criptomoedas (1-1000)",
    min_value=1,
    max_value=1000,
    value=1000,
    key="num_cryptos_table_input",
    help="Limita o número de criptomoedas exibidas na tabela 'Resultado da Consulta Multi-Nível' e usadas nos alertas (após os filtros, ordenado por Market Cap).",
)
st.sidebar.markdown("---")


# --- 3. Seleção Rápida de Colunas ---
st.sidebar.selectbox(
    "Seleção Rápida de Colunas",
    options=list(quick_select_options.keys()),
    key="quick_select_option",
    on_change=update_selected_keys,
)

# --- 4. Botões de Restaurar ---
st.sidebar.markdown("---")
st.sidebar.button(
    "🔄 Restaurar Colunas Default",
    on_click=restore_keys_callback,
    key="restore_keys_btn",
    help="Restaura a lista de colunas exibidas e filtráveis para a seleção padrão (Todas).",
)
st.sidebar.button(
    "🔄 Restaurar Filtros Default",
    on_click=restore_filters_callback,
    key="restore_filters_btn",
    help="Restaura todos os sliders de filtros numéricos para o valor mínimo e máximo do conjunto de dados.",
)
st.sidebar.markdown("---")

# --- 5. Multiselect de Colunas (Filtros e Colunas) ---
st.sidebar.markdown("### 📋 Seleção de Colunas")
multiselect_key = f"multiselect_{st.session_state.multiselect_key_counter}"
chaves_selecionadas = st.sidebar.multiselect(
    "Escolha colunas para consulta",
    options=[*df_total.columns],
    default=st.session_state.selected_keys,
    key=multiselect_key,
    on_change=multiselect_callback,
    help="Selecione as colunas que você deseja ver na tabela. Colunas numéricas selecionadas aparecerão na seção 'Filtros Numéricos' abaixo.",
)

st.sidebar.markdown("---")

# --- 6. Sliders e Number Inputs de Filtro ---
st.sidebar.markdown("### 🎚️ Filtros Numéricos")
for c in st.session_state.selected_keys:
    if c in df_total.columns and pd.api.types.is_numeric_dtype(df_total[c]):
        min_val_df = float(df_total[c].min())
        max_val_df = float(df_total[c].max())

        # Determinação dos limites e valores atuais do slider
        if c == "marketcap":
            min_val_df_limit = round(
                float(df_total[c].min()), 2
            )  # Limite mínimo absoluto do DF
            max_val_df_limit = round(float(df_total[c].max()) * 1.1, 2)
            widget_min_value = min_val_df_limit
            default_min_for_partial_update = marketcap_options["100M"]

        elif c != "price" and c != "dominance":
            min_val_df_limit = round(
                min_val_df * 0.9 if min_val_df > 0 else min_val_df * 1.1, 2
            )
            max_val_df_limit = round(
                max_val_df * 1.1 if max_val_df > 0 else max_val_df * 0.9, 2
            )
            widget_min_value = min_val_df_limit
            default_min_for_partial_update = min_val_df_limit

        elif c == "price":
            min_val_df_limit = 0.0
            # CORREÇÃO CRÍTICA: Aumenta a margem para 20% no price para evitar erros de float
            max_val_df_limit = round(df_total["price"].max() * 1.2, 2)
            widget_min_value = min_val_df_limit
            default_min_for_partial_update = 0.0

        elif c == "dominance":
            min_val_df_limit = 0.0
            max_val_df_limit = round(max_val_df * 1.1, 2)
            widget_min_value = min_val_df_limit
            default_min_for_partial_update = 0.0
        else:
            min_val_df_limit = round(min_val_df, 2)
            max_val_df_limit = round(max_val_df, 2)
            widget_min_value = min_val_df_limit
            default_min_for_partial_update = min_val_df_limit

        # Pega os valores atuais do slider (que já está no session_state/default)
        current_min_val, current_max_val = st.session_state.slider_values.get(
            c, (default_min_for_partial_update, max_val_df_limit)
        )

        # Garante que as chaves dos number_inputs existem
        if f"{c}_min" not in st.session_state:
            st.session_state[f"{c}_min"] = current_min_val
        if f"{c}_max" not in st.session_state:
            st.session_state[f"{c}_max"] = current_max_val

        # CRÍTICO: Forçar o valor do session_state a ser válido para o min/max do widget
        st.session_state[f"{c}_min"] = max(
            st.session_state[f"{c}_min"], widget_min_value
        )
        st.session_state[f"{c}_max"] = min(
            st.session_state[f"{c}_max"], max_val_df_limit
        )  # Adiciona a checagem de max_value

        def update_slider_from_inputs(key):
            """Atualiza o valor do slider no session_state a partir dos number_inputs."""
            min_input = st.session_state.get(f"{key}_min")
            max_input = st.session_state.get(f"{key}_max")

            # Recalcula limites/default min para atualizações parciais
            if key == "marketcap":
                max_val_df_ = round(float(df_total["marketcap"].max()) * 1.1, 2)
                default_min_for_partial_update_func = marketcap_options["100M"]
            elif key != "price" and key != "dominance":
                min_val_df_ = round(
                    float(
                        df_total[key].min() * 0.9
                        if float(df_total[key].min()) > 0
                        else float(df_total[key].min()) * 1.1
                    ),
                    2,
                )
                max_val_df_ = round(
                    float(
                        df_total[key].max() * 1.1
                        if float(df_total[key].max()) > 0
                        else float(df_total[key].max()) * 0.9
                    ),
                    2,
                )
                default_min_for_partial_update_func = min_val_df_
            elif key == "price":
                # Usa 1.2 na função
                max_val_df_ = round(df_total["price"].max() * 1.2, 2)
                default_min_for_partial_update_func = 0.0
            elif key == "dominance":
                max_val_df_ = round(df_total["dominance"].max() * 1.1, 2)
                default_min_for_partial_update_func = 0.0
            else:
                min_val_df_ = round(float(df_total[key].min()), 2)
                max_val_df_ = round(float(df_total[key].max()), 2)
                default_min_for_partial_update_func = min_val_df_

            # Atualiza o slider_values com os valores digitados
            if (
                min_input is not None
                and max_input is not None
                and min_input <= max_input
            ):
                st.session_state.slider_values[key] = (
                    float(min_input),
                    float(max_input),
                )
            elif min_input is not None and max_input is None:
                st.session_state.slider_values[key] = (float(min_input), max_val_df_)
            elif max_input is not None and min_input is None:
                st.session_state.slider_values[key] = (
                    default_min_for_partial_update_func,
                    float(max_input),
                )

        # O valor do slider é obtido diretamente dos valores do session_state
        selected_range = st.sidebar.slider(
            f"{c} ({human_format(widget_min_value)} - {human_format(max_val_df_limit)})",
            min_value=widget_min_value,
            max_value=max_val_df_limit,
            value=(st.session_state[f"{c}_min"], st.session_state[f"{c}_max"]),
            key=f"{c}_slider",
            # Atualiza o number input Mín e Máx ao mover o slider.
            on_change=lambda c=c: (
                st.session_state.slider_values.update(
                    {c: st.session_state[f"{c}_slider"]}
                ),
                st.session_state.update(
                    {f"{c}_min": st.session_state[f"{c}_slider"][0]}
                ),
                st.session_state.update(
                    {f"{c}_max": st.session_state[f"{c}_slider"][1]}
                ),
            ),
            help=f"Filtra a coluna '{c}' para valores dentro deste intervalo. A tabela e os alertas só mostrarão criptomoedas que atendam a este critério.",
        )

        # Number Input Mínimo
        st.sidebar.number_input(
            f"Min {c}",
            min_value=widget_min_value,
            max_value=max_val_df_limit,
            key=f"{c}_min",
            on_change=update_slider_from_inputs,
            args=(c,),
        )

        # Number Input Máximo
        st.sidebar.number_input(
            f"Max {c}",
            min_value=widget_min_value,
            max_value=max_val_df_limit,
            key=f"{c}_max",
            on_change=update_slider_from_inputs,
            args=(c,),
        )
        st.sidebar.markdown("---")

# ----------------------------
# Aplicação dos Filtros no DataFrame
# ----------------------------
df_filtrado = df_total.copy()
for c, (min_val, max_val) in st.session_state.slider_values.items():
    if c in df_filtrado.columns and pd.api.types.is_numeric_dtype(df_filtrado[c]):
        df_filtrado = df_filtrado[
            ((df_filtrado[c] >= min_val) & (df_filtrado[c] <= max_val))
            | df_filtrado[c].isnull()
        ]

# Aplica o limite de linhas da tabela (num_cryptos_table)
df_display = df_filtrado.copy()
if "marketcap" in df_display.columns:
    df_display = df_display.nlargest(
        int(st.session_state.num_cryptos_table_input), "marketcap", keep="all"
    )

# ----------------------------
# Resultado da Consulta Multi-Nível
# ----------------------------
st.markdown(
    f"## 🔎 Resultado da Consulta Multi-Nível (Top {len(df_display)} de {st.session_state.num_cryptos_table_input} Pedidas)"
)


renomear = {
    "performance.min15": "perf.min15",
    "performance.hour": "perf.hour",
    "performance.hour4": "perf.hour4",
    "performance.day": "perf.day",
    "performance.week": "perf.week",
    "performance.month": "perf.month",
    "performance.month3": "perf.month3",
    "performance.year": "perf.year",
    "marketcap": "Market Cap",
    "coingecko_links": "CGecko",
    "tradingview_links": "TView (F)",
    "tradingview_links_chart": "TView (C)",
}
df_display = df_display.rename(columns=renomear)

# Prioriza a ordem das default_keys, colocando novas chaves no final
ordered_cols = []
selected_set = set(st.session_state.selected_keys)
current_df_cols = set(df_display.columns)

# 1. Adiciona as colunas padrão que estão selecionadas (na ordem padrão)
for key in default_keys:
    if key in selected_set:
        renamed_key = renomear.get(key, key)
        if renamed_key in current_df_cols:
            ordered_cols.append(renamed_key)

# 2. Adiciona colunas que foram selecionadas, mas não estavam na lista padrão (novas)
for key in st.session_state.selected_keys:
    if key not in default_keys:
        renamed_key = renomear.get(key, key)
        if renamed_key in current_df_cols and renamed_key not in ordered_cols:
            ordered_cols.append(renamed_key)

# Remove duplicatas (apenas por garantia)
ordered_cols = list(dict.fromkeys(ordered_cols))

df_display_final = df_display[
    [col for col in ordered_cols if col in df_display.columns]
].copy()


if "dominance" in df_display_final.columns:
    df_display_final["dominance"] = (
        pd.to_numeric(df_display_final["dominance"], errors="coerce", downcast="float")
        * 100
    )

# Usar set para remover duplicações na lista de colunas a serem formatadas
perf_renomeadas = list(set([renomear.get(k, k) for k in perf_keys]))
column_config = {}

for col in perf_renomeadas:
    if col in df_display_final.columns:
        # Garante que a coluna é uma Series antes de chamar to_numeric
        df_display_final[col] = pd.to_numeric(
            df_display_final[col], errors="coerce", downcast="float"
        )

        if col == "perf.year":
            column_config[col] = st.column_config.NumberColumn(
                label="Year", format="%.2f%%", width=78
            )
        elif col == "perf.month3":
            column_config[col] = st.column_config.NumberColumn(
                label="Month3", format="%.2f%%", width=78
            )
        elif col == "perf.month":
            column_config[col] = st.column_config.NumberColumn(
                label="Month", format="%.2f%%", width=70
            )
        else:
            column_config[col] = st.column_config.NumberColumn(
                label=col.replace("perf.", "").capitalize(), format="%.2f%%"
            )

if "name" in df_display_final.columns:
    column_config["name"] = st.column_config.TextColumn(
        label="Name", pinned=True, width=120
    )
if "symbol" in df_display_final.columns:
    column_config["symbol"] = st.column_config.TextColumn(label="Symbol")
if "dominance" in df_display_final.columns:
    column_config["dominance"] = st.column_config.NumberColumn(
        label="Dominance", format="%.2f%%"
    )
if "price" in df_display_final.columns:
    df_display_final["price"] = pd.to_numeric(
        df_display_final["price"], errors="coerce", downcast="float"
    )
    column_config["price"] = st.column_config.NumberColumn(
        label="Price", format="$ %.8f"
    )
if "Market Cap" in df_display_final.columns:
    df_display_final["Market Cap"] = pd.to_numeric(
        df_display_final["Market Cap"], errors="coerce", downcast="float"
    )
    column_config["Market Cap"] = st.column_config.NumberColumn(
        label="Market Cap", format="compact", help="Capitalização de Mercado"
    )
if "volume" in df_display_final.columns:
    df_display_final["volume"] = pd.to_numeric(
        df_display_final["volume"], errors="coerce", downcast="float"
    )
    column_config["volume"] = st.column_config.NumberColumn(
        label="Volume", format="compact", help="Formato numérico sem formatação"
    )
if "links" in df_display_final.columns:
    column_config["links"] = st.column_config.LinkColumn(
        "CMCap", help="Link para o CoinMarketCap", display_text="Ver"
    )
if "CGecko" in df_display_final.columns:
    column_config["CGecko"] = st.column_config.LinkColumn(
        "CGecko", help="Link para o CoinGecko", display_text="Ver"
    )
if "TView (F)" in df_display_final.columns:
    column_config["TView (F)"] = st.column_config.LinkColumn(
        "TView (F)",
        help="Link para o Overview Rápido no TradingView",
        display_text="Fast",
    )
if "TView (C)" in df_display_final.columns:
    column_config["TView (C)"] = st.column_config.LinkColumn(
        "TView (C)",
        help="Link para o Gráfico Completo no TradingView",
        display_text="Chart",
    )

if not df_display_final.empty:
    valid_perf_cols = [
        col for col in perf_renomeadas if col in df_display_final.columns
    ]

    styled_df = df_display_final.style.map(
        color_performance,
        subset=valid_perf_cols,
    )

    st.dataframe(
        styled_df,
        height=500,
        column_config=column_config,
        hide_index=False,
    )
else:
    st.warning("⚠️ Nenhuma chave válida selecionada ou filtro muito restritivo.")

# ----------------------------
# Alertas Top 3 Performance
# ----------------------------
st.markdown("---")
# Garante que o df_alertas é baseado apenas nos índices filtrados
df_alertas = df_total.loc[df_display_final.index].copy()

# Adiciona as colunas de performance renomeadas ao df_alertas
for original_key, renamed_key in renomear.items():
    if (
        original_key in df_total.columns
        and renamed_key in df_display_final.columns
        and renamed_key not in df_alertas.columns
    ):
        df_alertas[renamed_key] = df_total[original_key].loc[df_alertas.index]
    # No caso de Market Cap e outras colunas que não são de performance mas foram renomeadas
    if (
        original_key != renamed_key
        and original_key in df_total.columns
        and renamed_key in df_display_final.columns
        and renamed_key not in df_alertas.columns
    ):
        df_alertas[renomeado_key] = df_total[original_key].loc[df_alertas.index]


# Mapeamento para as colunas de performance RENOMEADAS (chaves do df_alertas)
# E suas respectivas colunas de rankDiffs ORIGINAIS (valores do df_alertas)
intervalos = {
    "perf.hour": "Hora",
    "perf.day": "Dia",
    "perf.week": "Semana",
    "perf.month": "Mês",
    "perf.month3": "3 Meses",
}

rank_map_renomeado = {
    "perf.hour": "rankDiffs.hour",
    "perf.day": "rankDiffs.day",
    "perf.week": "rankDiffs.week",
    "perf.month": "rankDiffs.month",
    "perf.month3": "rankDiffs.month3",
}

st.markdown(
    f"## 🚨 Alertas Top 3 Performance (Top {len(df_alertas)} de {st.session_state.num_cryptos_table_input} Pedidas)"
)

for key_renomeada, label in intervalos.items():
    rank_key_original = rank_map_renomeado.get(key_renomeada)

    if (
        key_renomeada in df_alertas.columns
        and rank_key_original in df_alertas.columns
        and not df_alertas.empty
    ):
        df_alertas[key_renomeada] = pd.to_numeric(
            df_alertas[key_renomeada], errors="coerce"
        )

        # Garante que a coluna de performance existe e tem valores válidos antes de nlargest/nsmallest
        if df_alertas[key_renomeada].count() < 3:
            st.warning(
                f"⚠️ Dados insuficientes (menos de 3 moedas com valor) para alertas de {label}."
            )
            continue

        top_altas = df_alertas.nlargest(3, key_renomeada)
        top_baixas = df_alertas.nsmallest(3, key_renomeada)

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"### 🟢 Maiores Altas - {label}")
            for _, r in top_altas.iterrows():
                link_name_text = r.get("name", "N/A")
                if r.get("links"):
                    link_name_text = f"[{r.get('name', 'N/A')}]({r.get('links')})"

                symbol_display = f"**[{r.get('symbol', 'N/A')}]**"

                current_rank = r.get("rank", "N/A")
                rank_change = r.get(rank_key_original)
                rank_change_str = (
                    str(int(rank_change)) if not pd.isna(rank_change) else "N/A"
                )

                rank_info_text = f"[{current_rank} / {rank_change_str}]"
                if r.get("coingecko_links"):
                    rank_info = f"**[{rank_info_text}]({r.get('coingecko_links')})**"
                else:
                    rank_info = f"**{rank_info_text}**"

                price_text = f"${r.get('price', 0):.8f}"
                if r.get("tradingview_links"):
                    price_display = f"**[{price_text}]({r.get('tradingview_links')})**"
                else:
                    price_display = f"**{price_text}**"

                perf_text = f"{r.get(key_renomeada, 0):.2f}%"
                if r.get("tradingview_links_chart"):
                    perf_display = (
                        f"**[{perf_text}]({r.get('tradingview_links_chart')})**"
                    )
                else:
                    perf_display = f"**{perf_text}**"

                st.markdown(
                    f"- {link_name_text} {symbol_display} {rank_info} ({price_display} | {perf_display})"
                )
        with colB:
            st.markdown(f"### 🔴 Maiores Baixas - {label}")
            for _, r in top_baixas.iterrows():
                link_name_text = r.get("name", "N/A")
                if r.get("links"):
                    link_name_text = f"[{r.get('name', 'N/A')}]({r.get('links')})"

                symbol_display = f"**[{r.get('symbol', 'N/A')}]**"

                current_rank = r.get("rank", "N/A")
                rank_change = r.get(rank_key_original)
                rank_change_str = (
                    str(int(rank_change)) if not pd.isna(rank_change) else "N/A"
                )

                rank_info_text = f"[{current_rank} / {rank_change_str}]"
                if r.get("coingecko_links"):
                    rank_info = f"**[{rank_info_text}]({r.get('coingecko_links')})**"
                else:
                    rank_info = f"**{rank_info_text}**"

                price_text = f"${r.get('price', 0):.8f}"
                if r.get("tradingview_links"):
                    price_display = f"**[{price_text}]({r.get('tradingview_links')})**"
                else:
                    price_display = f"**{price_text}**"

                perf_text = f"{r.get(key_renomeada, 0):.2f}%"
                if r.get("tradingview_links_chart"):
                    perf_display = (
                        f"**[{perf_text}]({r.get('tradingview_links_chart')})**"
                    )
                else:
                    perf_display = f"**{perf_text}**"

                st.markdown(
                    f"- {link_name_text} {symbol_display} {rank_info} ({price_display} | {perf_display})"
                )
    else:
        st.warning(
            f"⚠️ Dados insuficientes para alertas de {label} (certifique-se de ter selecionado colunas de performance no filtro e que existam dados no subconjunto filtrado)."
        )

# ----------------------------
# Exportação
# ----------------------------
st.markdown("## 💾 Exportação de Dados")
if not df_display_final.empty:
    df_export = df_display_final.copy()
    cols_to_drop = [
        "links",
        "coingecko_links",
        "tradingview_links",
        "tradingview_links_chart",
    ]
    df_export = df_export.drop(
        columns=[col for col in cols_to_drop if col in df_export.columns]
    )

    dados_excel = converter_para_excel(df_export)
    st.download_button("📥 Baixar dados filtrados", dados_excel, "dados_filtrados.xlsx")

# ----------------------------
# Estatísticas de Desempenho
# ----------------------------
st.markdown("## 📊 Estatísticas de Desempenho (Top 1000 - Sem Stablecoins)")
if not df.empty:
    df_no_stablecoins = df[df["stable"] == False].copy()
    perf_keys_available = [k for k in perf_keys if k in df_no_stablecoins.columns]

    mean_values = (
        df_no_stablecoins[perf_keys_available].mean(numeric_only=True).round(2)
    )
    median_values = (
        df_no_stablecoins[perf_keys_available].median(numeric_only=True).round(2)
    )
    std_values = df_no_stablecoins[perf_keys_available].std(numeric_only=True).round(2)
    min_values = df_no_stablecoins[perf_keys_available].min(numeric_only=True).round(2)
    max_values = df_no_stablecoins[perf_keys_available].max(numeric_only=True).round(2)

    performance_labels_map = {
        "performance.min15": "Min15",
        "performance.hour": "Hour",
        "performance.hour4": "Hour4",
        "performance.day": "Day",
        "performance.week": "Week",
        "performance.month": "Month",
        "performance.month3": "Month3",
        "performance.year": "Year",
    }
    performance_labels = [performance_labels_map[k] for k in perf_keys_available]

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

    styled_stats_df = stats_df.style.map(
        lambda val: (
            "background-color: #228822; color: white"
            if float(val.strip("%")) > 0
            else (
                "background-color: #AA3333; color: white"
                if float(val.strip("%")) < 0
                else ""
            )
        ),
        subset=pd.IndexSlice[:, ["Média (%)", "Mediana (%)"]],
    )

    st.dataframe(styled_stats_df)
else:
    st.warning("⚠️ Não há dados disponíveis para calcular estatísticas.")

# ----------------------------
# Estatísticas de Desempenho Filtrada
# ----------------------------
st.markdown("---")
num_cryptos = st.number_input(
    "Número de criptomoedas a considerar (1-1000)",
    min_value=1,
    max_value=1000,
    value=100,
    key="num_cryptos_stats_input",
    help="Define o número de criptomoedas (ordenadas por Market Cap) usadas no cálculo das estatísticas abaixo. Não afeta a tabela principal.",
)

st.markdown(
    f"## 📊 Estatísticas de Desempenho (Top {st.session_state.num_cryptos_stats_input} - Sem Stablecoins)"
)
if not df_total.empty:
    df_no_stablecoins_filtered = df_total[df_total["stable"] == False].copy()

    df_top_cryptos = (
        df_no_stablecoins_filtered.nlargest(
            int(st.session_state.num_cryptos_stats_input), "marketcap", keep="all"
        )
        if "marketcap" in df_no_stablecoins_filtered.columns
        else df_no_stablecoins_filtered
    )
    perf_keys_available = [k for k in perf_keys if k in df_top_cryptos.columns]
    mean_values = df_top_cryptos[perf_keys_available].mean(numeric_only=True).round(2)
    median_values = (
        df_top_cryptos[perf_keys_available].median(numeric_only=True).round(2)
    )
    std_values = df_top_cryptos[perf_keys_available].std(numeric_only=True).round(2)
    min_values = df_top_cryptos[perf_keys_available].min(numeric_only=True).round(2)
    max_values = df_top_cryptos[perf_keys_available].max(numeric_only=True).round(2)

    performance_labels_map = {
        "performance.min15": "Min15",
        "performance.hour": "Hour",
        "performance.hour4": "Hour4",
        "performance.day": "Day",
        "performance.week": "Week",
        "performance.month": "Month",
        "performance.month3": "Month3",
        "performance.year": "Year",
    }
    performance_labels = [performance_labels_map[k] for k in perf_keys_available]

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

    styled_stats_df_filtered = stats_df_filtered.style.map(
        lambda val: (
            "background-color: #228822; color: white"
            if float(val.strip("%")) > 0
            else (
                "background-color: #AA3333; color: white"
                if float(val.strip("%")) < 0
                else ""
            )
        ),
        subset=pd.IndexSlice[:, ["Média (%)", "Mediana (%)"]],
    )

    st.dataframe(styled_stats_df_filtered)
else:
    st.warning("⚠️ Não há dados disponíveis para calcular estatísticas.")

# ----------------------------
# Loop de atualização automática
# ----------------------------
st.markdown("---")
st.write(f"Atualizando automaticamente em 120 segundos...")
time.sleep(120)
st.rerun()
