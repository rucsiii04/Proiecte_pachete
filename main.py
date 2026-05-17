import warnings

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Analiza rezervărilor hoteliere - Expedia")

# ------------------------------------------------------------
# LISTE DE COLOANE FOLOSITE PENTRU A NU TRATA ID-URILE GREȘIT
# ------------------------------------------------------------
ID_COLS = [
    "srch_id",
    "prop_id",
    "srch_destination_id",
    "site_id",
    "visitor_location_country_id",
    "position",
]

TARGET_COLS = ["booking_bool", "click_bool"]

BINARY_OR_DISCRETE_COLS = [
    "prop_brand_bool",
    "promotion_flag",
    "srch_saturday_night_bool",
    "random_bool",
    "booking_bool",
    "click_bool",
]

CONTINUOUS_ANALYSIS_COLS = [
    "price_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "orig_destination_distance",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "srch_length_of_stay",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
]

OUTLIER_COLS = [
    "price_usd",
    "prop_location_score1",
    "prop_location_score2",
    "srch_length_of_stay",
]

ENCODE_COLS = [
    "prop_starrating",
    "prop_brand_bool",
    "promotion_flag",
    "srch_saturday_night_bool",
    "random_bool",
]

CLUSTER_FEATURES = [
    "price_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
]

MODEL_FEATURES = [
    "price_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "srch_length_of_stay",
    "srch_adults_count",
    "promotion_flag",
]

OLS_FEATURES = [
    "prop_location_score1",
    "prop_location_score2",
    "prop_review_score",
    "prop_starrating",
]


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def numeric_analysis_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=["float64", "int64", "int32", "float32"]).columns.tolist()
    excluded = set(ID_COLS + TARGET_COLS)
    return [col for col in numeric_cols if col not in excluded]


def continuous_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in existing_cols(df, CONTINUOUS_ANALYSIS_COLS) if col not in ID_COLS + TARGET_COLS]


def current_base_df() -> pd.DataFrame:
    return st.session_state.get(
        "df_final_all",
        st.session_state.get("df_final", st.session_state.get("df_clean", st.session_state["df"])),
    )


def explain(text: str):
    st.info(text)



def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def top_missing_interpretation(missing_df: pd.DataFrame) -> str:
    nonzero = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    if nonzero.empty:
        return "Interpretare: în setul încărcat nu există valori lipsă, deci nu este necesară imputarea pentru această etapă."
    top = nonzero.head(5)
    parts = [f"`{idx}` are {row['Missing Count']:.0f} valori lipsă ({row['Missing %']:.2f}%)" for idx, row in top.iterrows()]
    return (
        "Interpretare: cele mai multe valori lipsă apar la " + "; ".join(parts) + ". "
        "Asta arată că unele informații nu sunt disponibile pentru multe sesiuni. În special, coloanele cu istoric al utilizatorului pot lipsi deoarece mulți utilizatori nu au istoric anterior în date."
    )


def describe_interpretation(df: pd.DataFrame, cols: list[str]) -> str:
    messages = []
    if "price_usd" in cols:
        price = df["price_usd"].dropna()
        if not price.empty:
            messages.append(
                f"Pentru price_usd, prețul mediu este {price.mean():.2f} USD, mediana este {price.median():.2f} USD, "
                f"iar valorile sunt între {price.min():.2f} și {price.max():.2f} USD. "
            )
    if "prop_review_score" in cols:
        rev = df["prop_review_score"].dropna()
        if not rev.empty:
            messages.append(
                f"Pentru prop_review_score, scorul mediu este {rev.mean():.2f} din 5. "
                f"Minimul este {rev.min():.2f}, iar maximul este {rev.max():.2f}, deci datele permit compararea calității percepute a hotelurilor."
            )
    if "prop_starrating" in cols:
        stars = df["prop_starrating"].dropna()
        if not stars.empty:
            messages.append(
                f"Pentru prop_starrating, media este {stars.mean():.2f} stele, cu valori între {stars.min():.0f} și {stars.max():.0f}. "
                "Aceasta descrie nivelul general al hotelurilor prezente în eșantion."
            )
    if "booking_bool" in df.columns:
        rate = df["booking_bool"].mean() * 100
        messages.append(
            f"Rata de rezervare (booking_bool = 1) este {rate:.2f}%. "
            "Această valoare fiind una mică, arată că setul de date este dezechilibrat: majoritatea căutărilor nu se finalizează cu rezervare."
        )
    if not messages:
        return "Interpretare: statisticile descriptive sintetizează valorile numerice relevante, fără a interpreta ID-urile ca variabile economice."
    return "Interpretare: " + "\n\n".join(messages)


def histogram_interpretation(df: pd.DataFrame, col: str) -> str:
    x = df[col].dropna()
    if x.empty:
        return f"Interpretare: coloana `{col}` nu are valori disponibile pentru histogramă."
    mean = x.mean(); median = x.median(); mn = x.min(); mx = x.max()
    q1 = x.quantile(0.25); q3 = x.quantile(0.75); iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    out_pct = ((x > upper).mean() * 100) if iqr != 0 else 0
    skew_text = "media este mai mare decât mediana, deci există valori mari care trag distribuția spre dreapta" if mean > median else "media este apropiată de mediană sau mai mică, deci distribuția nu este puternic trasă de valori foarte mari"
    return (
        f"Interpretare pentru `{col}`: media este {mean:.2f}, mediana este {median:.2f}, iar valorile merg de la {mn:.2f} la {mx:.2f}. "
        f"În aceste date, {skew_text}. Aproximativ {out_pct:.2f}% dintre valori sunt peste limita superioară IQR ({upper:.2f}), ceea ce indică posibile valori extreme."
    )


def boxplot_interpretation(df: pd.DataFrame, cols: list[str]) -> str:
    lines = []
    for col in cols:
        x = df[col].dropna()
        if x.empty:
            continue
        q1 = x.quantile(0.25); q3 = x.quantile(0.75); iqr = q3 - q1
        lower = q1 - 1.5 * iqr; upper = q3 + 1.5 * iqr
        out = ((x < lower) | (x > upper)).sum()
        lines.append(f"`{col}` are {out} valori extreme ({out / len(x) * 100:.2f}%), cu interval IQR acceptat între {lower:.2f} și {upper:.2f}")
    if not lines:
        return "Interpretare: boxplot-ul nu are suficiente valori pentru a evidenția concret valori extreme."
    return "Interpretare: " + "; ".join(lines) + ". Aceste valori pot influența media, scalarea și modelele, deci trebuie analizate separat."


def outlier_interpretation(df: pd.DataFrame, col: str, lower: float, upper: float, outliers: pd.DataFrame) -> str:
    total = len(df)
    pct = len(outliers) / total * 100 if total else 0
    return (
        f"Interpretare pentru `{col}`: limita inferioară calculată prin IQR este {lower:.2f}, iar limita superioară este {upper:.2f}. "
        f"Au fost găsite {len(outliers)} valori extreme din {total} rânduri, adică {pct:.2f}%. "
    )


def price_category_interpretation(cat_group: pd.DataFrame, counts: pd.Series, q33: float, q66: float) -> str:
    review_map = dict(zip(cat_group["Categorie preț"], cat_group["Review mediu"]))
    low = review_map.get("low"); high = review_map.get("high")
    parts = [f"Pragurile împart prețurile astfel: low până la {q33:.2f} USD, medium până la {q66:.2f} USD, high peste {q66:.2f} USD."]
    parts.append("Distribuția claselor este: " + ", ".join([f"{idx}: {val} înregistrări" for idx, val in counts.items()]) + ".")
    if low is not None and high is not None:
        diff = high - low
        if diff > 0:
            parts.append(f"Categoria high are review mediu mai mare decât categoria low cu {diff:.2f} puncte, ceea ce sugerează că hotelurile mai scumpe sunt evaluate mai bine în aceste date.")
        elif diff < 0:
            parts.append(f"Categoria high are review mediu mai mic decât categoria low cu {abs(diff):.2f} puncte, deci prețul mai mare nu este însoțit aici de review-uri mai bune.")
        else:
            parts.append("Categoria high și categoria low au același review mediu, deci nu apare o diferență vizibilă între preț și review în această grupare.")
    return "Interpretare: " + " ".join(parts)


def group_count_interpretation(group_count: pd.DataFrame, col: str, total: int) -> str:
    if group_count.empty:
        return "Interpretare: nu există valori de interpretat pentru această grupare."
    top = group_count.iloc[0]
    label = top.get("label", top[col])
    pct = top["count"] / total * 100 if total else 0
    return (
        f"Interpretare: cea mai frecventă valoare pentru `{col}` este {label}, cu {int(top['count'])} apariții, adică {pct:.2f}% din setul analizat. "
    )


def mean_destination_interpretation(group_mean: pd.DataFrame) -> str:
    if group_mean.empty:
        return "Interpretare: nu există destinații pentru calculul prețului mediu."
    top = group_mean.iloc[0]; bottom = group_mean.iloc[-1]
    return (
        f"Interpretare: destinația cu cel mai mare preț mediu este {top['destination_name']}, cu {top['price_usd']:.2f} USD. "
        f"Destinația cu cel mai mic preț mediu din acest tabel este {bottom['destination_name']}, cu {bottom['price_usd']:.2f} USD. "
        "Diferența arată că în date există zone cu niveluri de preț foarte diferite."
    )


def sum_destination_interpretation(group_sum: pd.DataFrame) -> str:
    if group_sum.empty:
        return "Interpretare: nu există destinații pentru suma prețurilor."
    top = group_sum.iloc[0]
    return (
        f"Interpretare: cea mai mare sumă a prețurilor apare la {top['destination_name']}, cu total {top['price_usd']:.2f} USD. "
        "Această valoare nu înseamnă automat că destinația este cea mai scumpă; ea combină prețurile cu numărul de apariții din setul de date."
    )


def agg_destination_interpretation(group_agg: pd.DataFrame) -> str:
    if group_agg.empty:
        return "Interpretare: nu există rezultate pentru agregarea multiplă."
    temp = group_agg.copy(); temp["range"] = temp["max"] - temp["min"]
    widest = temp.sort_values("range", ascending=False).iloc[0]
    top_mean = temp.sort_values("mean", ascending=False).iloc[0]
    return (
        f"Interpretare: cel mai mare preț mediu este în {top_mean['destination_name']} ({top_mean['mean']:.2f} USD). "
        f"Cea mai mare diferență între preț minim și maxim apare în {widest['destination_name']}, unde intervalul este de la {widest['min']:.2f} la {widest['max']:.2f} USD. "
        "Un interval mare indică o destinație cu oferte variate, de la hoteluri ieftine la hoteluri scumpe."
    )


def hotel_stats_interpretation(hotel_stats_sorted: pd.DataFrame) -> str:
    if hotel_stats_sorted.empty:
        return "Interpretare: nu există hoteluri pentru calculul statisticilor."
    top = hotel_stats_sorted.iloc[0]
    return (
        f"Interpretare: hotelul care apare cel mai des este {top['hotel_name']}, cu {int(top['nr_aparitii'])} apariții. "
        f"Prețul său mediu este {top['pret_mediu']:.2f} USD, iar prețul maxim este {top['pret_maxim']:.2f} USD. "
        "Un număr mare de apariții indică faptul că hotelul este prezent frecvent în căutările analizate."
    )


def combo_interpretation(top_dest: pd.DataFrame) -> str:
    if top_dest.empty:
        return "Interpretare: nu există combinații destinație-hotel pentru analiză."
    top = top_dest.iloc[0]
    return (
        f"Interpretare: cea mai frecventă combinație este {top['destination_name']} - {top['hotel_name']}, cu {int(top['nr_aparitii'])} apariții și preț mediu {top['pret_mediu']:.2f} USD. "
        "Aceasta arată unde apare cel mai des un hotel într-o anumită destinație."
    )


def overpriced_interpretation(overpriced: pd.DataFrame) -> str:
    if overpriced.empty:
        return "Interpretare: nu au fost identificate hoteluri cu preț mediu peste media destinației."
    top = overpriced.iloc[0]
    return (
        f"Interpretare: cea mai mare diferență față de media destinației apare pentru {top['hotel_name']} din {top['destination_name']}. "
        f"Hotelul are preț mediu {top['hotel_mean_price']:.2f} USD, media destinației este {top['dest_mean_price']:.2f} USD, diferența fiind {top['price_diff']:.2f} USD. "
        "Diferența foarte mare poate apărea și deoarece hotelul are un număr redus de apariții în setul de date, iar câteva valori foarte ridicate influențează puternic media. De asemenea, prezența outlierilor poate crește artificial prețul mediu al hotelului."
    )


def scaling_interpretation(stats: pd.DataFrame, cols: list[str]) -> str:
    if not cols:
        return "Interpretare: nu au existat coloane potrivite pentru scalare."
    sample_cols = cols[:5]
    checks = []
    for col in sample_cols:
        if col in stats.columns:
            checks.append(f"`{col}` are media {stats.loc['mean', col]:.4f} și deviația standard {stats.loc['std', col]:.4f}")
    return (
        "Interpretare: după StandardScaler, variabilele scalate trebuie să aibă media aproximativ 0 și deviația standard aproximativ 1. "
        + ("În rezultatele tale, " + "; ".join(checks) + ". " if checks else "")
        + "Acest pas este important pentru KMeans și modele sensibile la scară, deoarece prețul nu mai domină automat variabilele cu valori mai mici."
    )


def kmeans_interpretation(counts: pd.Series, sil_score) -> str:
    if counts.empty:
        return "Interpretare: nu există clustere rezultate."
    total = counts.sum(); biggest = counts.idxmax(); smallest = counts.idxmin()
    text = (
        f"Interpretare: cel mai mare cluster este clusterul {int(biggest)}, cu {int(counts.loc[biggest])} observații ({counts.loc[biggest] / total * 100:.2f}%). "
        f"Cel mai mic cluster este clusterul {int(smallest)}, cu {int(counts.loc[smallest])} observații ({counts.loc[smallest] / total * 100:.2f}%). "
    )
    if sil_score is not None:
        if sil_score < 0.25:
            quality = "separarea clusterelor este slabă, deci grupurile nu sunt foarte clar delimitate"
        elif sil_score < 0.5:
            quality = "separarea clusterelor este moderată, deci există o anumită structură, dar nu foarte puternică"
        else:
            quality = "separarea clusterelor este bună, deci grupurile sunt relativ bine delimitate"
        text += f"Silhouette Score este {sil_score:.4f}; {quality}. "
    text += "Clusterele sunt interpretate ca segmente de hoteluri pe baza prețului, stelelor, review-ului și scorurilor de locație."
    return text


def logreg_interpretation(acc: float, report_dict: dict, target_counts: dict) -> str:
    total = sum(target_counts.values()) if target_counts else 0
    positive = target_counts.get(1, target_counts.get(1.0, target_counts.get('1', 0)))
    rate = positive / total * 100 if total else 0
    cls1 = report_dict.get('1', report_dict.get('1.0', {}))
    precision = cls1.get('precision', 0); recall = cls1.get('recall', 0); f1 = cls1.get('f1-score', 0)
    return (
        f"Interpretare: rezervările reale reprezintă doar {rate:.2f}% din observații, ceea ce arată că setul de date este puternic dezechilibrat. "

        f"Modelul obține o acuratețe de {acc:.4f}, însă această valoare nu este suficientă pentru evaluare deoarece majoritatea sesiunilor nu se finalizează cu rezervare. "

        f"Recall={recall:.4f} arată că modelul identifică aproximativ {recall * 100:.2f}% dintre rezervările reale. "

        f"Precision={precision:.4f} arată că multe dintre rezervările prezise de model sunt de fapt alarme false. "

        f"F1-score={f1:.4f} confirmă că modelul are dificultăți în separarea corectă a rezervărilor reale de sesiunile fără rezervare."
    )


def ols_interpretation(model) -> str:
    r2 = model.rsquared
    params = model.params.drop(labels=['const'], errors='ignore')
    if not params.empty:
        strongest = params.abs().sort_values(ascending=False).index[0]
        coef = params[strongest]
        direction = "pozitivă" if coef > 0 else "negativă"
        coef_text = f"Cel mai mare coeficient ca valoare absolută este pentru `{strongest}` ({coef:.4f}), cu influență {direction} asupra prețului scalat."
    else:
        coef_text = "Nu există coeficienți explicativi în afara constantei."
    return (
        f"Interpretare: R-squared este {r2:.4f}, deci modelul explică aproximativ {r2 * 100:.2f}% din variația prețului în forma scalată. "
        f"{coef_text} Coeficienții trebuie citiți împreună cu P>|t|: dacă p-value este mare, efectul nu este clar statistic."
    )


def compare_interpretation(dt_results: dict, rf_results: dict) -> str:
    dt_auc = dt_results['roc_auc']; rf_auc = rf_results['roc_auc']
    better = "Random Forest" if rf_auc > dt_auc else "Decision Tree"
    dt_rep = dt_results.get('report_dict', {})
    rf_rep = rf_results.get('report_dict', {})
    dt_recall = dt_rep.get('1', dt_rep.get('1.0', {})).get('recall', 0)
    rf_recall = rf_rep.get('1', rf_rep.get('1.0', {})).get('recall', 0)
    return (
        f"Interpretare: Decision Tree obține un ROC-AUC de {dt_auc:.4f}, iar Random Forest {rf_auc:.4f}. "
        f"În această analiză, Random Forest separă puțin mai bine rezervările reale de sesiunile fără rezervare. "

        f"Recall-ul pentru clasa 1 este {dt_recall:.4f} la Decision Tree și {rf_recall:.4f} la Random Forest. "
        "Această metrică arată cât de multe rezervări reale reușește să identifice fiecare model. "

        "În cazul acestui set de date dezechilibrat, ROC-AUC și recall sunt mai importante decât acuratețea, "
        "deoarece obiectivul principal este identificarea rezervărilor reale, nu doar clasificarea corectă a clasei majoritare fără rezervare. \n"
        "În concluzie, Random Forest are performanțe ușor mai bune decât Decision Tree pe acest set de date, însă ambele modele au rezultate slabe în identificarea rezervărilor reale."
    )


def safe_train_test_split(X: pd.DataFrame, y: pd.Series):
    # Stratify merge doar dacă există cel puțin două clase și fiecare clasă are minimum 2 observații.
    counts = y.value_counts()
    stratify = y if len(counts) >= 2 and counts.min() >= 2 else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)


# --------------------------- SIDEBAR ---------------------------
st.sidebar.title("Navigație")
st.sidebar.markdown(
    """
- [Introducere](#intro)
- [Încărcare date](#upload)
- [Vizualizare date](#view)
- [EDA](#eda)
- [Curățare date](#clean)
- [Prelucrări pe date](#groupby)
- [Preprocesare și modele](#prep)
- [Comparare algoritmi](#compare)
""",
    unsafe_allow_html=True,
)

# --------------------------- INTRO ---------------------------
st.markdown("<a id='intro'></a>", unsafe_allow_html=True)
st.header("Introducere")
explain(
    """
Scopul aplicației este analiza comportamentului utilizatorilor și identificarea factorilor
care influențează rezervările hoteliere pe platforma Expedia.
"""
)

# --------------------------- UPLOAD ---------------------------
st.markdown("<a id='upload'></a>", unsafe_allow_html=True)
st.header("Încărcare date")
uploaded_file = st.file_uploader("Încarcă fișier CSV", type=["csv"])

if uploaded_file is not None:
    file_signature = (uploaded_file.name, uploaded_file.size)

    if st.session_state.get("uploaded_file_signature") != file_signature:
        df = pd.read_csv(uploaded_file, low_memory=False)
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)

        st.session_state["df"] = df
        st.session_state["uploaded_file_signature"] = file_signature


        for key in [
            "df_clean",
            "df_final",
            "df_final_all",
            "df_encoded",
            "df_scaled",
            "df_cluster",
            "missing_msg",
            "outliers_msg",
            "outliers_all_msg",
            "encoding_msg",
            "scaling_msg",
            "kmeans_msg",
            "logreg_msg",
            "ols_msg",
            "compare_msg",
        ]:
            st.session_state.pop(key, None)

        st.success("Set de date încărcat cu succes!")
    else:
        st.info("Setul de date este deja încărcat.")

# --------------------------- VIEW ---------------------------
st.markdown("<a id='view'></a>", unsafe_allow_html=True)
st.header("Vizualizare date")

if "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("Primele 5 rânduri")
    st.dataframe(df.head())
    st.write("Dimensiune set de date:", df.shape)
else:
    st.warning("Încarcă mai întâi un set de date.")

# --------------------------- EDA ---------------------------
st.markdown("<a id='eda'></a>", unsafe_allow_html=True)
st.header("Exploratory Data Analysis (EDA)")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Tipul de date pentru o coloană")
    selected_any_col = st.selectbox("Alege o coloană:", df.columns.tolist(), key="dtype_col")
    st.write(f"Tipul de date pentru coloana `{selected_any_col}` este: {df[selected_any_col].dtype}")
    explain(
        "Verificarea tipurilor de date arată dacă o coloană poate fi analizată numeric, "
        "dacă este text sau dacă este doar un identificator."
    )

    st.subheader("Statistici descriptive pentru variabile relevante")
    desc_cols = numeric_analysis_cols(df)
    if desc_cols:
        st.dataframe(df[desc_cols].describe().T)
        explain(describe_interpretation(df, desc_cols))
    else:
        st.write("Nu există variabile numerice relevante pentru statistici descriptive.")

    st.subheader("Coloane de tip identificator")
    present_ids = existing_cols(df, ID_COLS)
    if present_ids:
        st.write(present_ids)
        explain(
            "Aceste coloane identifică sesiuni, hoteluri sau destinații. Ele pot fi folosite pentru numărare și grupare, "
            "dar nu sunt analizate ca valori numerice propriu-zise."
        )
    else:
        st.write("Nu au fost identificate coloane ID din lista folosită în aplicație.")

    st.subheader("Coloane categorice/text")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        for col in categorical_cols[:5]:
            st.write(f"{col} - primele valori frecvente:")
            st.write(df[col].value_counts().head(10))
        explain(
            "Coloanele text sunt analizate prin frecvențe. În acest set de date, coloanele text sunt puține, "
            "iar multe informații sunt deja codificate numeric."
        )
    else:
        st.write("Nu există coloane categorice de tip text.")

    st.subheader("Valori lipsă")
    missing_count = df.isnull().sum()
    missing_percent = ((missing_count / len(df)) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_percent})
    st.dataframe(missing_df.sort_values("Missing %", ascending=False))
    st.bar_chart(missing_count[missing_count > 0])
    explain(top_missing_interpretation(missing_df))

    st.subheader("Histogramă pentru variabile numerice relevante")
    hist_cols = continuous_cols(df)
    if hist_cols:
        selected_col = st.selectbox("Alege coloană numerică relevantă", hist_cols, key="hist_col")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[selected_col].dropna(), bins=30, edgecolor="black")
        ax.set_title(f"Distribuția pentru {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frecvență")
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        explain(histogram_interpretation(df, selected_col))
    else:
        st.write("Nu există coloane numerice relevante pentru histogramă.")
else:
    st.warning("Încarcă mai întâi un set de date pentru EDA.")

# --------------------------- CURATAARE DATE ---------------------------
st.markdown("<a id='clean'></a>", unsafe_allow_html=True)
st.header("Curățarea datelor")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Tratarea valorilor lipsă")
    st.write(
        "Se aplică imputarea cu mediana pentru coloanele numerice relevante. "
        "ID-urile nu sunt imputate, deoarece sunt identificatori tehnici."
    )

    if st.button("Aplică tratarea valorilor lipsă"):
        df_clean = df.copy()
        fill_cols = numeric_analysis_cols(df_clean)
        for col in fill_cols:
            if df_clean[col].isna().any():
                median_value = df_clean[col].median()
                if pd.notna(median_value):
                    df_clean[col] = df_clean[col].fillna(median_value)

        st.session_state["df_clean"] = df_clean
        st.session_state["missing_msg"] = "Valorile lipsă numerice relevante au fost imputate cu mediana."
        st.session_state["missing_after"] = df_clean.isnull().sum()
        st.session_state["missing_fill_cols"] = fill_cols

    if "missing_msg" in st.session_state:
        st.success(st.session_state["missing_msg"])
        st.write("Coloane analizate pentru imputare:", st.session_state["missing_fill_cols"])
        missing_after = st.session_state["missing_after"]
        if missing_after.sum() == 0:
            st.success("Nu mai există valori lipsă în set de date.")
        else:
            st.write("Valori lipsă rămase:")
            st.dataframe(missing_after[missing_after > 0])
        filled_cols = st.session_state["missing_fill_cols"]
        remaining = missing_after[missing_after > 0].sort_values(ascending=False)
        if remaining.empty:
            explain(
                "Interpretare: după imputare nu mai există valori lipsă în set de date. Valorile numerice lipsă au fost completate cu mediana coloanei, ceea ce este potrivit când există valori extreme."
            )
        else:
            explain(
                "Interpretare: au fost analizate pentru imputare coloanele " + ", ".join(filled_cols) + ". "
                "După imputare, încă rămân valori lipsă în coloane care nu au fost completate automat, de exemplu: "
                + ", ".join([f"{idx}: {val}" for idx, val in remaining.head(5).items()]) + "."
            )

    df_used = st.session_state.get("df_clean", df)

    st.subheader("Boxplot pentru variabile numerice relevante")
    box_cols = continuous_cols(df_used)
    default_box_cols = existing_cols(df_used, ["price_usd", "prop_review_score", "prop_location_score1"])
    if box_cols:
        selected_box_cols = st.multiselect(
            "Alege coloane pentru boxplot",
            box_cols,
            default=default_box_cols if default_box_cols else box_cols[: min(3, len(box_cols))],
            key="box_cols",
        )
        if selected_box_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot([df_used[col].dropna() for col in selected_box_cols], labels=selected_box_cols)
            ax.set_title("Distribuția și valorile extreme pentru variabile relevante")
            ax.set_xticklabels(selected_box_cols, rotation=45)
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            explain(boxplot_interpretation(df_used, selected_box_cols))
        else:
            st.write("Selectează cel puțin o coloană.")
    else:
        st.write("Nu există coloane numerice relevante pentru boxplot.")

    st.subheader("Detectare valori extreme (IQR)")
    outlier_options = existing_cols(df_used, OUTLIER_COLS)
    if outlier_options:
        selected_col_outliers = st.selectbox(
            "Alege coloană pentru outliers",
            outlier_options,
            key="outlier_col",
        )
        Q1 = df_used[selected_col_outliers].quantile(0.25)
        Q3 = df_used[selected_col_outliers].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_used[
            (df_used[selected_col_outliers] < lower_bound)
            | (df_used[selected_col_outliers] > upper_bound)
        ]

        st.write(f"Număr outliers: {len(outliers)}")
        st.write(f"Procent outliers: {(len(outliers) / len(df_used)) * 100:.2f}%")
        st.dataframe(outliers.head())
        explain(outlier_interpretation(df_used, selected_col_outliers, lower_bound, upper_bound, outliers))

        if st.button("Elimină outliers pentru coloana selectată"):
            df_no_outliers = df_used[
                (df_used[selected_col_outliers] >= lower_bound)
                & (df_used[selected_col_outliers] <= upper_bound)
            ]
            st.session_state["df_final"] = df_no_outliers
            st.session_state["outliers_msg"] = f"Outliers eliminați pentru coloana {selected_col_outliers}."
            st.session_state["outliers_removed"] = len(df_used) - len(df_no_outliers)
            st.session_state["outliers_shape"] = df_no_outliers.shape

        if "outliers_msg" in st.session_state:
            st.success(st.session_state["outliers_msg"])
            st.write("Număr valori eliminate:", st.session_state["outliers_removed"])
            st.write("Dimensiune nouă:", st.session_state["outliers_shape"])
    else:
        st.write("Nu există coloane potrivite pentru detectarea outlierilor.")

    st.subheader("Eliminare outliers pentru setul de coloane relevante")
    cols_outliers = existing_cols(df_used, OUTLIER_COLS)
    if cols_outliers:
        if st.button("Elimină outliers din coloanele relevante"):
            df_no_outliers_all = df_used.copy()
            initial_shape = df_no_outliers_all.shape
            mask = pd.Series([True] * len(df_no_outliers_all), index=df_no_outliers_all.index)

            for col in cols_outliers:
                Q1 = df_no_outliers_all[col].quantile(0.25)
                Q3 = df_no_outliers_all[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask = mask & (df_no_outliers_all[col] >= lower_bound) & (df_no_outliers_all[col] <= upper_bound)

            df_no_outliers_all = df_no_outliers_all[mask]
            st.session_state["df_final_all"] = df_no_outliers_all
            st.session_state["outliers_all_msg"] = "Outliers eliminați din coloanele numerice continue relevante."
            st.session_state["outliers_all_cols"] = cols_outliers
            st.session_state["outliers_all_removed"] = initial_shape[0] - df_no_outliers_all.shape[0]
            st.session_state["outliers_all_shape"] = df_no_outliers_all.shape

        if "outliers_all_msg" in st.session_state:
            st.success(st.session_state["outliers_all_msg"])
            st.write("Coloane analizate:", st.session_state["outliers_all_cols"])
            st.write("Număr rânduri eliminate:", st.session_state["outliers_all_removed"])
            st.write("Dimensiune nouă:", st.session_state["outliers_all_shape"])
            removed_pct = st.session_state["outliers_all_removed"] / (st.session_state["outliers_all_removed"] + st.session_state["outliers_all_shape"][0]) * 100
            explain(
                f"Interpretare: au fost eliminate {st.session_state['outliers_all_removed']} rânduri, adică aproximativ {removed_pct:.2f}% din datele analizate. "
                f"Filtrarea s-a făcut pe {st.session_state['outliers_all_cols']}. ID-urile nu au fost folosite la eliminare, deoarece nu pot fi outlieri economici."
            )

    if "df_final" in st.session_state:
        st.subheader("Date după curățare pe coloana selectată")
        st.dataframe(st.session_state["df_final"].head())
    if "df_final_all" in st.session_state:
        st.subheader("Date după curățare pe coloanele relevante")
        st.dataframe(st.session_state["df_final_all"].head())
else:
    st.warning("Încarcă mai întâi un set de date.")

# --------------------------- GROUPBY + AGREGARE ---------------------------
st.markdown("<a id='groupby'></a>", unsafe_allow_html=True)
st.header("Analiză prin grupare și agregare")

if "df" in st.session_state:
    df = st.session_state.get("df_final_all", st.session_state.get("df_final", st.session_state.get("df_clean", st.session_state["df"]))).copy()

    st.subheader("Creare variabilă categorică din variabilă numerică")
    st.write("Transformăm `price_usd` în 3 clase: low / medium / high, pe baza cuantilelor.")
    if "price_usd" in df.columns:
        q33 = df["price_usd"].quantile(0.33)
        q66 = df["price_usd"].quantile(0.66)

        def categorize_price(price):
            if price <= q33:
                return "low"
            if price <= q66:
                return "medium"
            return "high"

        df["price_category"] = df["price_usd"].apply(categorize_price)
        st.session_state["df_with_price_category"] = df

        st.write(f"Praguri: low ≤ {q33:.2f} | medium ≤ {q66:.2f} | high > {q66:.2f}")
        st.write("Distribuție clase:")
        st.write(df["price_category"].value_counts())

        cat_group = df.groupby("price_category")["prop_review_score"].agg(["mean", "count"]).reset_index()
        cat_group.columns = ["Categorie preț", "Review mediu", "Nr. înregistrări"]
        st.dataframe(cat_group)
        st.bar_chart(cat_group.set_index("Categorie preț")["Review mediu"])
        explain(price_category_interpretation(cat_group, df["price_category"].value_counts(), q33, q66))
    else:
        st.warning("Coloana price_usd nu există.")

    st.subheader("Problema 1: Număr înregistrări per categorie")
    group_options = existing_cols(
        df,
        ["srch_destination_id", "prop_id", "promotion_flag", "booking_bool", "prop_starrating"],
    )
    if group_options:
        col = st.selectbox("Alege coloană pentru grupare", group_options, key="p1")
        group_count = df.groupby(col).size().reset_index(name="count")
        if col == "prop_id":
            group_count["label"] = "Hotel " + group_count["prop_id"].astype(str)
        elif col == "srch_destination_id":
            group_count["label"] = "Zona " + group_count["srch_destination_id"].astype(str)
        else:
            group_count["label"] = group_count[col].astype(str)
        group_count = group_count.sort_values("count", ascending=False)
        st.dataframe(group_count.head(20))
        st.bar_chart(group_count.set_index("label")["count"].head(20))
        explain(group_count_interpretation(group_count, col, len(df)))
    else:
        st.warning("Nu există coloane potrivite pentru grupare.")

    st.subheader("Problema 2: Preț mediu pe destinație")
    if "srch_destination_id" in df.columns and "price_usd" in df.columns:
        group_mean = df.groupby("srch_destination_id")["price_usd"].mean().reset_index()
        group_mean["destination_name"] = "Zona " + group_mean["srch_destination_id"].astype(str)
        group_mean = group_mean.sort_values("price_usd", ascending=False)
        st.dataframe(group_mean[["destination_name", "price_usd"]].head(20))
        st.bar_chart(group_mean.set_index("destination_name")["price_usd"].head(20))
        explain(mean_destination_interpretation(group_mean))
    else:
        st.warning("Coloanele necesare nu există.")

    st.subheader("Problema 3: Suma prețurilor pe destinație")
    if "srch_destination_id" in df.columns and "price_usd" in df.columns:
        group_sum = df.groupby("srch_destination_id")["price_usd"].sum().reset_index()
        group_sum["destination_name"] = "Zona " + group_sum["srch_destination_id"].astype(str)
        group_sum = group_sum.sort_values("price_usd", ascending=False)
        st.dataframe(group_sum[["destination_name", "price_usd"]].head(20))
        explain(sum_destination_interpretation(group_sum))
    else:
        st.warning("Coloanele necesare nu există.")

    st.subheader("Problema 4: Agregare multiplă pe destinație")
    if "srch_destination_id" in df.columns and "price_usd" in df.columns:
        group_agg = df.groupby("srch_destination_id")["price_usd"].agg(["mean", "min", "max"]).reset_index()
        group_agg["destination_name"] = "Zona " + group_agg["srch_destination_id"].astype(str)
        group_agg = group_agg.sort_values("mean", ascending=False)
        st.dataframe(group_agg[["destination_name", "mean", "min", "max"]].head(20))
        explain(agg_destination_interpretation(group_agg))
    else:
        st.warning("Coloanele necesare nu există.")

    st.subheader("Problema 5: Top hoteluri după performanță")
    if "prop_id" in df.columns and "price_usd" in df.columns:
        hotel_stats = df.groupby("prop_id").agg({"price_usd": ["count", "mean", "max"]})
        hotel_stats.columns = ["nr_aparitii", "pret_mediu", "pret_maxim"]
        hotel_stats = hotel_stats.reset_index()
        hotel_stats["hotel_name"] = "Hotel " + hotel_stats["prop_id"].astype(str)
        hotel_stats_sorted = hotel_stats.sort_values(by=["nr_aparitii", "pret_mediu"], ascending=[False, False])
        st.dataframe(hotel_stats_sorted[["hotel_name", "nr_aparitii", "pret_mediu", "pret_maxim"]].head(20))
        explain(hotel_stats_interpretation(hotel_stats_sorted))
    else:
        st.warning("Coloanele necesare nu există.")

    st.subheader("Problema 6: Analiză pe destinație și hotel")
    if "srch_destination_id" in df.columns and "prop_id" in df.columns and "price_usd" in df.columns:
        multi_group = df.groupby(["srch_destination_id", "prop_id"]).agg({"price_usd": ["count", "mean"]})
        multi_group.columns = ["nr_aparitii", "pret_mediu"]
        multi_group = multi_group.reset_index()
        multi_group["hotel_name"] = "Hotel " + multi_group["prop_id"].astype(str)
        multi_group["destination_name"] = "Zona " + multi_group["srch_destination_id"].astype(str)
        top_dest = multi_group.sort_values(by="nr_aparitii", ascending=False)
        st.dataframe(top_dest[["destination_name", "hotel_name", "nr_aparitii", "pret_mediu"]].head(20))
        explain(combo_interpretation(top_dest))
    else:
        st.warning("Coloanele necesare nu există.")

    st.subheader("Problema 7: Identificarea hotelurilor supraevaluate")
    if "srch_destination_id" in df.columns and "prop_id" in df.columns and "price_usd" in df.columns:
        dest_mean = df.groupby("srch_destination_id")["price_usd"].mean().reset_index()
        dest_mean = dest_mean.rename(columns={"price_usd": "dest_mean_price"})
        hotel_mean = df.groupby(["srch_destination_id", "prop_id"])["price_usd"].mean().reset_index()
        hotel_mean = hotel_mean.rename(columns={"price_usd": "hotel_mean_price"})
        merged = pd.merge(hotel_mean, dest_mean, on="srch_destination_id")
        merged["price_diff"] = merged["hotel_mean_price"] - merged["dest_mean_price"]
        overpriced = merged[merged["price_diff"] > 0].copy()
        overpriced["hotel_name"] = "Hotel " + overpriced["prop_id"].astype(str)
        overpriced["destination_name"] = "Zona " + overpriced["srch_destination_id"].astype(str)
        overpriced = overpriced.sort_values(by="price_diff", ascending=False)
        st.dataframe(overpriced[["destination_name", "hotel_name", "hotel_mean_price", "dest_mean_price", "price_diff"]].head(20))

        top10 = overpriced.head(10)
        if not top10.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = "Hotel " + top10["prop_id"].astype(str)
            ax.bar(labels, top10["price_diff"])
            ax.set_title("Top hoteluri peste media destinației")
            ax.set_xlabel("Hotel")
            ax.set_ylabel("Diferență preț")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        explain(overpriced_interpretation(overpriced))
    else:
        st.warning("Coloanele necesare nu există.")
else:
    st.warning("Nu există date încă. Te rog încarcă fișierul CSV.")

# --------------------------- PREPROCESARE SI MODELE ---------------------------
st.markdown("<a id='prep'></a>", unsafe_allow_html=True)
st.header("Preprocesare și modele")

if "df" in st.session_state:
    df_base = current_base_df().copy()

    st.subheader("Codificarea variabilelor categorice")
    cols_to_encode = existing_cols(df_base, ENCODE_COLS)
    df_encoded = df_base.copy()
    st.write("Coloane selectate pentru Label Encoding:", cols_to_encode)
    if st.button("Aplică Label Encoding"):
        mappings = {}
        for col in cols_to_encode:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            mappings[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
        st.session_state["df_encoded"] = df_encoded
        st.session_state["encoding_msg"] = "Codificarea a fost aplicată."
        st.session_state["encoding_cols"] = cols_to_encode
        st.session_state["encoding_mappings"] = mappings

    if "encoding_msg" in st.session_state:
        st.success(st.session_state["encoding_msg"])
        for col, mapping in st.session_state["encoding_mappings"].items():
            st.write(f"Mapping `{col}`: {mapping}")
        if st.session_state["encoding_cols"]:
            st.dataframe(st.session_state["df_encoded"][st.session_state["encoding_cols"]].head())
        explain(
            "Interpretare: au fost codificate următoarele coloane: " + ", ".join(st.session_state["encoding_cols"]) + ". "
            "Pentru variabilele binare, codificarea păstrează logica 0/1. Pentru `prop_starrating`, ordinea numerică rămâne interpretabilă deoarece numărul de stele are sens ordinal."
        )

    st.subheader("Scalarea variabilelor numerice")
    df_scale_input = st.session_state.get("df_encoded", df_base).copy()
    scale_cols = numeric_analysis_cols(df_scale_input)
    scale_cols = [col for col in scale_cols if col not in BINARY_OR_DISCRETE_COLS]
    st.write("Coloane numerice folosite pentru scaling:", scale_cols)

    if scale_cols:
        if st.button("Aplică Standard Scaling"):
            scaler = StandardScaler()
            df_scaled = df_scale_input.copy()
            df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
            st.session_state["df_scaled"] = df_scaled
            st.session_state["scaling_msg"] = "Scalarea a fost aplicată."
            st.session_state["scaling_cols"] = scale_cols
            st.session_state["scaling_stats"] = df_scaled[scale_cols].describe()

        if "scaling_msg" in st.session_state:
            st.success(st.session_state["scaling_msg"])
            st.write("Coloane scalate:", st.session_state["scaling_cols"])
            st.dataframe(st.session_state["scaling_stats"])
            explain(
                "Prin aplicarea StandardScaler, toate variabilele selectate sunt aduse pe aceeași scară statistică (medie aproximativ 0 și deviație standard aproximativ 1), astfel încât fiecare variabilă să contribuie mai echilibrat la model."

                "Coloanele de tip ID, precum prop_id sau srch_id, nu sunt scalate deoarece ele reprezintă doar identificatori tehnici și nu au semnificație numerică reală. De asemenea, variabilele țintă booking_bool și click_bool sunt păstrate în forma originală pentru a putea interpreta corect rezultatele modelelor predictive."
            )
    else:
        st.write("Nu există coloane numerice potrivite pentru scaling.")

    if "df_scaled" in st.session_state:
        st.subheader("Set de date final după preprocesare")
        st.dataframe(st.session_state["df_scaled"].head())
        st.write("Dimensiune set de date:", st.session_state["df_scaled"].shape)

    st.subheader("Clusterizare (KMeans)")
    if "df_scaled" in st.session_state:
        df_cluster = st.session_state["df_scaled"].copy()
        cluster_features = existing_cols(df_cluster, CLUSTER_FEATURES)
        st.write("Features folosite pentru clustering:", cluster_features)
        if len(cluster_features) >= 2:
            k = st.slider("Alege numărul de clustere (k)", 2, 6, 3)
            if st.button("Aplică KMeans"):
                df_cluster_clean = df_cluster[cluster_features].dropna()
                if len(df_cluster_clean) < k:
                    st.error("Nu există suficiente rânduri curate pentru numărul de clustere ales.")
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(df_cluster_clean)
                    df_cluster.loc[df_cluster_clean.index, "cluster"] = labels

                    # Silhouette pe eșantion, ca să nu crape aplicația pe set de date mare.
                    sample_size = min(3000, len(df_cluster_clean))
                    sil_score = None
                    if sample_size > k and len(set(labels)) > 1:
                        sample_X = df_cluster_clean.sample(n=sample_size, random_state=42)
                        labels_series = pd.Series(labels, index=df_cluster_clean.index)
                        sample_labels = labels_series.loc[sample_X.index]
                        sil_score = silhouette_score(sample_X, sample_labels)

                    st.session_state["df_cluster"] = df_cluster
                    st.session_state["kmeans_msg"] = "Clusterizare realizată."
                    st.session_state["kmeans_k"] = k
                    st.session_state["kmeans_features"] = cluster_features
                    st.session_state["kmeans_counts"] = df_cluster["cluster"].value_counts().sort_index()
                    st.session_state["kmeans_silhouette"] = sil_score

            if "kmeans_msg" in st.session_state:
                st.success(st.session_state["kmeans_msg"])
                st.write("Număr clustere:", st.session_state["kmeans_k"])
                st.write("Număr elemente per cluster:")
                st.write(st.session_state["kmeans_counts"])
                if st.session_state["kmeans_silhouette"] is not None:
                    st.write(f"Silhouette Score calculat pe eșantion: {st.session_state['kmeans_silhouette']:.4f}")
                else:
                    st.write("Silhouette Score nu a putut fi calculat în siguranță pentru această configurație.")
                st.dataframe(st.session_state["df_cluster"][st.session_state["kmeans_features"] + ["cluster"]].head(10))
                explain(kmeans_interpretation(st.session_state["kmeans_counts"], st.session_state["kmeans_silhouette"]))
        else:
            st.warning("Nu există suficiente coloane pentru KMeans.")
    else:
        st.warning("Aplică mai întâi Standard Scaling.")

    st.subheader("Regresie Logistică")
    st.write("Scopul este prezicerea variabilei `booking_bool`: 1 = rezervare, 0 = fără rezervare.")
    if "df_scaled" in st.session_state:
        df_model = st.session_state["df_scaled"].copy()
        target = "booking_bool"
        logreg_features = existing_cols(df_model, MODEL_FEATURES)
        if target in df_model.columns and logreg_features:
            df_model_clean = df_model[logreg_features + [target]].dropna()
            X = df_model_clean[logreg_features]
            y = df_model_clean[target]
            st.write(f"Distribuție target: {y.value_counts().to_dict()}")
            if y.nunique() < 2:
                st.warning("Targetul are o singură clasă după curățare, deci modelul nu poate fi antrenat.")
            elif st.button("Antrenează model Logistic Regression"):
                X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
                model = LogisticRegression(max_iter=1000, class_weight="balanced")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, zero_division=0)
                report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                st.session_state["logreg_msg"] = "Model Logistic Regression antrenat."
                st.session_state["logreg_acc"] = acc
                st.session_state["logreg_report"] = report
                st.session_state["logreg_report_dict"] = report_dict
                st.session_state["logreg_target_counts"] = y.value_counts().to_dict()

            if "logreg_msg" in st.session_state:
                st.success(st.session_state["logreg_msg"])
                st.write(f"Acuratețe: {st.session_state['logreg_acc']:.4f}")
                st.text(st.session_state["logreg_report"])
                explain(logreg_interpretation(
                    st.session_state["logreg_acc"],
                    st.session_state["logreg_report_dict"],
                    st.session_state["logreg_target_counts"],
                ))
        else:
            st.warning("Nu există suficiente coloane pentru regresia logistică.")

    st.subheader("Regresie multiplă (StatsModels OLS)")
    st.write("Scopul este modelarea prețului hotelului (`price_usd`) în funcție de caracteristici ale proprietății.")
    if "df_scaled" in st.session_state:
        df_reg = st.session_state["df_scaled"].copy()
        reg_features = existing_cols(df_reg, OLS_FEATURES)
        target = "price_usd"
        if len(reg_features) >= 1 and target in df_reg.columns:
            df_reg_clean = df_reg[reg_features + [target]].dropna()
            if len(df_reg_clean) >= len(reg_features) + 2:
                X = sm.add_constant(df_reg_clean[reg_features])
                y = df_reg_clean[target]
                if st.button("Rulează regresia multiplă"):
                    model = sm.OLS(y, X).fit()
                    st.session_state["ols_msg"] = "Modelul OLS a fost antrenat."
                    st.session_state["ols_summary"] = model.summary().as_text()
                    st.session_state["ols_r2"] = model.rsquared
                    st.session_state["ols_params"] = model.params

                if "ols_msg" in st.session_state:
                    st.success(st.session_state["ols_msg"])
                    st.text(st.session_state["ols_summary"])
                    r2 = st.session_state["ols_r2"]
                    params = st.session_state["ols_params"].drop(labels=["const"], errors="ignore")
                    if not params.empty:
                        strongest = params.abs().sort_values(ascending=False).index[0]
                        coef = params[strongest]
                        direction = "pozitivă" if coef > 0 else "negativă"
                        coef_text = f"Cel mai mare coeficient ca valoare absolută este pentru `{strongest}` ({coef:.4f}), cu influență {direction} asupra prețului scalat."
                    else:
                        coef_text = "Nu există coeficienți explicativi în afara constantei."
                    explain(
                        f"Interpretare: modelul obține un R-squared de {r2:.4f}, ceea ce înseamnă că aproximativ "
                        f"{r2 * 100:.2f}% din variația prețului este explicată de variabilele folosite în regresie. "
                        f"{coef_text} "
                        "Coeficienții trebuie interpretați împreună cu valorile P>|t|. "
                        "Dacă p-value este mare, influența variabilei asupra prețului nu este suficient de semnificativă statistic."
                    )
            else:
                st.warning("Nu există suficiente rânduri curate pentru OLS.")
        else:
            st.warning("Nu există suficiente coloane pentru regresie.")
else:
    st.warning("Încarcă mai întâi un set de date.")

# --------------------------- COMPARARE ALGORITMI ---------------------------
st.markdown("<a id='compare'></a>", unsafe_allow_html=True)
st.header("Comparare algoritmi: Decision Tree vs Random Forest")

if "df_scaled" in st.session_state:
    df_model = st.session_state["df_scaled"].copy()
    target = "booking_bool"
    features = existing_cols(df_model, MODEL_FEATURES)

    if target in df_model.columns and features:
        df_model_clean = df_model[features + [target]].dropna()
        X = df_model_clean[features]
        y = df_model_clean[target]
        st.write(f"Features: {features}")
        st.write(f"Target: {target} | Distribuție: {y.value_counts().to_dict()}")

        if y.nunique() < 2:
            st.warning("Targetul are o singură clasă după curățare, deci modelele nu pot fi comparate.")
        elif st.button("Rulează comparația algoritmilor"):
            X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

            dt = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=8)
            dt.fit(X_train, y_train)
            y_pred_dt = dt.predict(X_test)
            y_proba_dt = dt.predict_proba(X_test)[:, 1]

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                max_depth=12,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            y_proba_rf = rf.predict_proba(X_test)[:, 1]

            st.session_state["compare_msg"] = "Comparația algoritmilor a fost realizată."
            st.session_state["dt_results"] = {
                "accuracy": accuracy_score(y_test, y_pred_dt),
                "roc_auc": roc_auc_score(y_test, y_proba_dt),
                "confusion_matrix": confusion_matrix(y_test, y_pred_dt),
                "report": classification_report(y_test, y_pred_dt, zero_division=0),
                "report_dict": classification_report(y_test, y_pred_dt, zero_division=0, output_dict=True),
            }
            st.session_state["rf_results"] = {
                "accuracy": accuracy_score(y_test, y_pred_rf),
                "roc_auc": roc_auc_score(y_test, y_proba_rf),
                "confusion_matrix": confusion_matrix(y_test, y_pred_rf),
                "report": classification_report(y_test, y_pred_rf, zero_division=0),
                "report_dict": classification_report(y_test, y_pred_rf, zero_division=0, output_dict=True),
            }

        if "compare_msg" in st.session_state:
            st.success(st.session_state["compare_msg"])
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Decision Tree")
                st.write("Accuracy:", st.session_state["dt_results"]["accuracy"])
                st.write("ROC-AUC:", st.session_state["dt_results"]["roc_auc"])
                st.write("Confusion Matrix:")
                st.write(st.session_state["dt_results"]["confusion_matrix"])
                st.text(st.session_state["dt_results"]["report"])
            with col2:
                st.subheader("Random Forest")
                st.write("Accuracy:", st.session_state["rf_results"]["accuracy"])
                st.write("ROC-AUC:", st.session_state["rf_results"]["roc_auc"])
                st.write("Confusion Matrix:")
                st.write(st.session_state["rf_results"]["confusion_matrix"])
                st.text(st.session_state["rf_results"]["report"])
            explain(compare_interpretation(st.session_state["dt_results"], st.session_state["rf_results"]))
    else:
        st.warning("Nu există suficiente coloane pentru compararea algoritmilor.")
else:
    st.warning("Aplică mai întâi pașii de preprocesare și scaling pentru a compara modelele.")
