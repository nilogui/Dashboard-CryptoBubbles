from io import BytesIO
import pandas as pd
import requests
import streamlit as st
import time

URL = "https://cryptobubbles.net/backend/data/bubbles1000.usd.json"

# ----------------------------
# Funções auxiliares
# ----------------------------


@st.cache_data(ttl=60)
def obter_dados(url=URL):
    """Obtém os dados do Cryptobubbles."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def normalizar_json(dados):
    """Normaliza o JSON e adiciona a coluna de rank."""
    df = pd.json_normalize(dados, sep=".")
    df["rank"] = df["marketcap"].rank(method="min", ascending=False).astype(int)
    return df


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


def update_selected_keys():
    selected_option = st.session_state.quick_select_option
    st.session_state.selected_keys = quick_select_options[selected_option]
    st.session_state.multiselect_key_counter += 1


def restore_keys_callback():
    st.session_state.selected_keys = default_keys.copy()
    st.session_state.multiselect_key_counter += 1


def restore_filters_callback():
    for c in default_keys:
        if c in df_total.columns and pd.api.types.is_numeric_dtype(df_total[c]):
            min_val = float(df_total[c].min())
            max_val = float(df_total[c].max())
            if c == "marketcap":
                min_val = 100_000_000.0
                max_val = max_val * 1.1
            elif c != "price" and c != "dominance":
                min_val = min_val * 0.9 if min_val > 0 else min_val * 1.1
                max_val = max_val * 1.1 if max_val > 0 else max_val * 0.9
            elif c == "dominance":
                min_val = 0.0
                max_val = max_val * 1.1
            st.session_state.slider_values[c] = (min_val, max_val)
    if "price" in df_total.columns:
        max_price = df_total["price"].max() * 1.1
        st.session_state.slider_values["price"] = (0.0, max_price)

    st.session_state.update_inputs = True


# --- 1. Marketcap Mínimo (Seleção Rápida) ---
# Necessário para definir a borda inferior do filtro Marketcap.
selected_min_marketcap = marketcap_options[
    "100M"
]  # Valor padrão, usado para inicialização
if "marketcap" in st.session_state.selected_keys or "marketcap" in df_total.columns:
    marketcap_quick_select = st.sidebar.selectbox(
        "Marketcap Mínimo (Seleção Rápida)",
        options=list(marketcap_options.keys()),
        index=list(marketcap_options.keys()).index("100M"),
    )
    selected_min_marketcap = marketcap_options[marketcap_quick_select]
    if "marketcap" not in st.session_state.slider_values:
        max_val_marketcap = float(df_total["marketcap"].max()) * 1.1
        st.session_state.slider_values["marketcap"] = (
            selected_min_marketcap,
            max_val_marketcap,
        )
    else:
        current_max = st.session_state.slider_values["marketcap"][1]
        st.session_state.slider_values["marketcap"] = (
            selected_min_marketcap,
            current_max,
        )
    st.sidebar.markdown("---")

# --- NOVO: 2. Número de criptomoedas na tabela (Input Numérico) ---
# Este valor será usado para limitar o DataFrame da tabela de exibição.
num_cryptos_table = st.sidebar.number_input(
    "Número de criptomoedas na tabela (1-1000)",
    min_value=1,
    max_value=1000,
    value=1000,  # VALOR ALTERADO PARA 1000
    key="num_cryptos_table_input",
    help="Limita o número de criptomoedas exibidas na tabela 'Resultado da Consulta Multi-Nível' (após os filtros, ordenado por Market Cap).",
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
)
st.sidebar.button(
    "🔄 Restaurar Filtros Default",
    on_click=restore_filters_callback,
    key="restore_filters_btn",
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
)
st.session_state.selected_keys = chaves_selecionadas

st.sidebar.markdown("---")

# --- 6. Sliders e Number Inputs de Filtro ---
st.sidebar.markdown("### 🎚️ Filtros Numéricos")
for c in chaves_selecionadas:
    if c in df_total.columns and pd.api.types.is_numeric_dtype(df_total[c]):
        min_val_df = float(df_total[c].min())
        max_val_df = float(df_total[c].max())

        if c == "marketcap":
            min_val_df_limit = float(df_total[c].min())
            max_val_df_limit = float(df_total[c].max()) * 1.1
            current_min_val = float(
                st.session_state.slider_values.get(
                    c, (selected_min_marketcap, max_val_df_limit)
                )[0]
            )
            current_max_val = float(
                st.session_state.slider_values.get(
                    c, (selected_min_marketcap, max_val_df_limit)
                )[1]
            )
        elif c != "price" and c != "dominance":
            min_val_df_limit = min_val_df * 0.9 if min_val_df > 0 else min_val_df * 1.1
            max_val_df_limit = max_val_df * 1.1 if max_val_df > 0 else max_val_df * 0.9
            current_min_val, current_max_val = st.session_state.slider_values.get(
                c, (min_val_df_limit, max_val_df_limit)
            )
        elif c == "price":
            min_val_df_limit = 0.0
            max_val_df_limit = df_total["price"].max() * 1.1
            current_min_val, current_max_val = st.session_state.slider_values.get(
                c, (min_val_df_limit, max_val_df_limit)
            )
        elif c == "dominance":
            min_val_df_limit = 0.0
            max_val_df_limit = max_val_df * 1.1
            current_min_val, current_max_val = st.session_state.slider_values.get(
                c, (min_val_df_limit, max_val_df_limit)
            )
        else:
            min_val_df_limit = min_val_df
            max_val_df_limit = max_val_df
            current_min_val, current_max_val = st.session_state.slider_values.get(
                c, (min_val_df_limit, max_val_df_limit)
            )

        def update_slider_from_inputs(key):
            min_input = st.session_state.get(f"{key}_min")
            max_input = st.session_state.get(f"{key}_max")

            # Recalcula limites para consistência com as regras de filtro
            if key == "marketcap":
                min_val_df_ = 100_000_000.0
                max_val_df_ = float(df_total["marketcap"].max()) * 1.1
            elif key != "price" and key != "dominance":
                min_val_df_ = (
                    float(df_total[key].min()) * 0.9
                    if float(df_total[key].min()) > 0
                    else float(df_total[key].min()) * 1.1
                )
                max_val_df_ = (
                    float(df_total[key].max()) * 1.1
                    if float(df_total[key].max()) > 0
                    else float(df_total[key].max()) * 0.9
                )
            elif key == "price":
                min_val_df_ = 0.0
                max_val_df_ = df_total["price"].max() * 1.1
            elif key == "dominance":
                min_val_df_ = 0.0
                max_val_df_ = df_total["dominance"].max() * 1.1
            else:
                min_val_df_ = float(df_total[key].min())
                max_val_df_ = float(df_total[key].max())

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
                st.session_state.slider_values[key] = (min_val_df_, float(max_input))

        selected_range = st.sidebar.slider(
            f"{c} ({human_format(min_val_df_limit)} - {human_format(max_val_df_limit)})",
            min_value=min_val_df_limit,
            max_value=max_val_df_limit,
            value=(current_min_val, current_max_val),
            key=f"{c}_slider",
            on_change=lambda c=c: st.session_state.slider_values.update(
                {c: st.session_state[f"{c}_slider"]}
            ),
        )

        st.sidebar.number_input(
            f"Min {c}",
            min_value=min_val_df_limit,
            max_value=max_val_df_limit,
            value=max(min_val_df_limit, selected_range[0]),
            key=f"{c}_min",
            on_change=update_slider_from_inputs,
            args=(c,),
        )
        st.sidebar.number_input(
            f"Max {c}",
            min_value=min_val_df_limit,
            max_value=max_val_df_limit,
            value=min(max_val_df_limit, selected_range[1]),
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
    df_display = df_display.nlargest(int(num_cryptos_table), "marketcap", keep="all")

# ----------------------------
# Resultado da Consulta Multi-Nível
# ----------------------------
st.markdown(
    f"## 🔎 Resultado da Consulta Multi-Nível (Top {len(df_display)} de {num_cryptos_table} Pedidas)"
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
    "coingecko_links": "CoinGecko",
    "tradingview_links": "TView (F)",
    "tradingview_links_chart": "TView (C)",
}
df_display = df_display.rename(columns=renomear)

ordered_cols = []
perf_renomeadas = [renomear.get(k, k) for k in perf_keys]
fixed_keys = ["name", "symbol", "price", "marketcap", "volume", "dominance"]

for key in ["name", "symbol", "price", "marketcap", "volume", "dominance"]:
    renamed_key = renomear.get(key, key)
    if key in st.session_state.selected_keys and renamed_key in df_display.columns:
        ordered_cols.append(renamed_key)

for key in perf_keys:
    if key in st.session_state.selected_keys:
        renamed_key = renomear.get(key)
        if renamed_key and renamed_key in df_display.columns:
            ordered_cols.append(renamed_key)

if "links" in st.session_state.selected_keys and "links" in df_display.columns:
    ordered_cols.append("links")
if (
    "coingecko_links" in st.session_state.selected_keys
    and "CoinGecko" in df_display.columns
):
    ordered_cols.append("CoinGecko")

if (
    "tradingview_links" in st.session_state.selected_keys
    and "TView (F)" in df_display.columns
):
    ordered_cols.append("TView (F)")
if (
    "tradingview_links_chart" in st.session_state.selected_keys
    and "TView (C)" in df_display.columns
):
    ordered_cols.append("TView (C)")

for key in st.session_state.selected_keys:
    if (
        key not in ["name", "symbol", "price", "marketcap", "volume", "dominance"]
        and key not in perf_keys
        and key
        not in [
            "links",
            "coingecko_links",
            "tradingview_links",
            "tradingview_links_chart",
        ]
        and key in df_display.columns
    ):
        ordered_cols.append(key)

ordered_cols = list(dict.fromkeys(ordered_cols))

df_display_final = df_display[
    [col for col in ordered_cols if col in df_display.columns]
].copy()

if "dominance" in df_display_final.columns:
    df_display_final["dominance"] = (
        pd.to_numeric(df_display_final["dominance"], errors="coerce", downcast="float")
        * 100
    )

column_config = {}
for col in perf_renomeadas:
    if col in df_display_final.columns:
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
        "CoinMarketCap", help="Link para o CoinMarketCap", display_text="Ver"
    )
if "CoinGecko" in df_display_final.columns:
    column_config["CoinGecko"] = st.column_config.LinkColumn(
        "CoinGecko", help="Link para o CoinGecko", display_text="Ver"
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
st.markdown("## 🚨 Alertas Top 3 Performance")
intervalos = {
    "performance.hour": "Hora",
    "performance.day": "Dia",
    "performance.week": "Semana",
    "performance.month": "Mês",
    "performance.month3": "3 Meses",
}

rank_map = {
    "performance.hour": "rankDiffs.hour",
    "performance.day": "rankDiffs.day",
    "performance.week": "rankDiffs.week",
    "performance.month": "rankDiffs.month",
    "performance.month3": "rankDiffs.month3",
}

colunas_essenciais = [
    "name",
    "slug",
    "symbol",
    "price",
    "rank",
    "links",
    "coingecko_links",
    "tradingview_links",
    "tradingview_links_chart",
] + list(rank_map.values())
colunas_disponiveis = [col for col in colunas_essenciais if col in df_filtrado.columns]

for key, label in intervalos.items():
    rank_key = rank_map.get(key)

    # Nota: df_filtrado (sem limite de linhas) é usado aqui para que os alertas usem todos os dados filtrados
    if (
        key in df_filtrado.columns
        and rank_key in df_filtrado.columns
        and not df_filtrado.empty
    ):
        df_filtrado[key] = pd.to_numeric(df_filtrado[key], errors="coerce")

        cols_para_selecao = [
            c for c in colunas_disponiveis if c not in [rank_key, key]
        ] + [rank_key, key]

        top_altas = df_filtrado.nlargest(3, key)[cols_para_selecao]
        top_baixas = df_filtrado.nsmallest(3, key)[cols_para_selecao]
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"### 🟢 Maiores Altas - {label}")
            for _, r in top_altas.iterrows():
                link_name_text = r.get("name", "N/A")
                if r.get("links"):
                    link_name_text = f"[{r.get('name', 'N/A')}]({r.get('links')})"

                symbol_display = f"**[{r.get('symbol', 'N/A')}]**"

                current_rank = r.get("rank", "N/A")
                rank_change = r.get(rank_key)
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

                perf_text = f"{r.get(key, 0):.2f}%"
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
                rank_change = r.get(rank_key)
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

                perf_text = f"{r.get(key, 0):.2f}%"
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
            f"⚠️ Dados insuficientes para alertas de {label} (faltando {key} ou {rank_key})."
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

    performance_labels = [
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
# Estatísticas de Desempenho Filtrada (Reintroduzindo o controle aqui)
# ----------------------------
st.markdown("---")
# O controle foi movido de volta para esta seção
num_cryptos = st.number_input(
    "Número de criptomoedas a considerar (1-1000)",
    min_value=1,
    max_value=1000,
    value=100,
    key="num_cryptos_stats_input",
    help="Define o número de criptomoedas (ordenadas por Market Cap) usadas no cálculo das estatísticas abaixo. Não afeta a tabela principal.",
)

st.markdown(f"## 📊 Estatísticas de Desempenho (Top {num_cryptos} - Sem Stablecoins)")
if not df_total.empty:
    df_no_stablecoins_filtered = df_total[df_total["stable"] == False].copy()

    df_top_cryptos = (
        df_no_stablecoins_filtered.nlargest(int(num_cryptos), "marketcap", keep="all")
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

    performance_labels = [
        "Min15",
        "Hour",
        "Hour4",
        "Day",
        "Week",
        "Month",
        "Month3",
        "Year",
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
st.write(f"Atualizando automaticamente em 60 segundos...")
time.sleep(60)
st.rerun()
