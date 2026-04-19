import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(layout="wide")

st.title("Analiza rezervărilor hoteliere - Expedia")

st.sidebar.title("Navigație")

st.sidebar.markdown("""
- [Introducere](#intro)
- [Încărcare date](#upload)
- [Vizualizare date](#view)
- [EDA](#eda)
- [Curățare date](#clean)
- [Prelucrări pe date](#groupby)
- [Preprocesare](#prep)
""", unsafe_allow_html=True)

st.markdown("<a id='intro'></a>", unsafe_allow_html=True)
st.header("Introducere")

st.write("""
Scopul aplicației este analiza comportamentului utilizatorilor
și identificarea factorilor care influențează rezervările hoteliere.
""")

st.markdown("<a id='upload'></a>", unsafe_allow_html=True)
st.header("Încărcare date")

uploaded_file = st.file_uploader("Încarcă fișier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.success("Dataset încărcat cu succes!")

st.markdown("<a id='view'></a>", unsafe_allow_html=True)
st.header("Vizualizare date")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Primele 5 rânduri")
    st.dataframe(df.head())
else:
    st.warning("Încarcă mai întâi un dataset.")

st.markdown("<a id='eda'></a>", unsafe_allow_html=True)
st.header("Exploratory Data Analysis (EDA)")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Preview date")
    st.dataframe(df.head())

    st.subheader("Dimensiune dataset")
    st.write(df.shape)
    st.subheader("Tipul de date pentru o coloană")
    coloane = df.columns.tolist()
    coloana_selectata = st.selectbox("Alege o coloană:", coloane)
    if coloana_selectata:
        tip_data = df[coloana_selectata].dtype
        st.write(f"Tipul de date pentru coloana '{coloana_selectata}' este: {tip_data}")
    st.subheader("Statistici descriptive")
    st.write(df.describe(include="all"))
    st.subheader("Coloane categorice")
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        for col in categorical_cols[:5]:
            st.write(f"{col} - valori unice:")
            st.write(df[col].value_counts().head(10))
    else:
        st.write("Nu există coloane categorice.")

    st.subheader("Valori lipsă")
    missing_count = df.isnull().sum()
    missing_percent = ((missing_count / len(df)) * 100).round(2)

    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_percent
    })

    st.dataframe(missing_df)
    st.bar_chart(missing_count)

    st.subheader("Histogramă")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Alege coloană numerică", numeric_cols, key="hist_col")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[selected_col].dropna(), bins=30, edgecolor="black")
        ax.set_title(f"Distribuția pentru {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frecvență")
        ax.grid(True)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Nu există coloane numerice.")
else:
    st.warning("Încarcă mai întâi un dataset pentru EDA.")

st.markdown("<a id='clean'></a>", unsafe_allow_html=True)
st.header("Curățarea datelor")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Tratarea valorilor lipsă")
    st.write("Se aplică imputarea cu **mediana** pentru toate coloanele numerice, "
             "deoarece distribuția variabilelor (ex: price_usd) este asimetrică și "
             "mediana este robustă la valorile extreme.")

    if st.button("Aplică tratarea valorilor lipsă"):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=["float64", "int64"]).columns

        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        st.session_state["df_clean"] = df_clean
        st.success("Valorile lipsă au fost imputate cu mediana.")
        st.dataframe(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

    df_used = st.session_state.get("df_clean", df)

    df_display = df_used.copy()

    df_display["hotel_name"] = "Hotel " + df_display["prop_id"].astype(str)
    df_display["destination_name"] = "Zona " + df_display["srch_destination_id"].astype(str)

    st.subheader("Boxplot pentru variabile numerice")
    numeric_cols = df_used.select_dtypes(include=["float64", "int64"]).columns

    if len(numeric_cols) > 0:
        selected_box_cols = st.multiselect(
            "Alege coloane pentru boxplot",
            numeric_cols,
            default=list(numeric_cols[:5]),
            key="box_cols"
        )

        if len(selected_box_cols) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot([df_used[col].dropna() for col in selected_box_cols], labels=selected_box_cols)
            ax.set_title("Distribuția și outliers pentru variabile numerice")
            ax.set_xticklabels(selected_box_cols, rotation=45)
            ax.grid(True)

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("Selectează cel puțin o coloană.")
    else:
        st.write("Nu există coloane numerice pentru boxplot.")

    st.subheader("Detectare valori extreme (IQR)")

    if len(numeric_cols) > 0:
        selected_col_outliers = st.selectbox(
            "Alege coloană pentru outliers",
            numeric_cols,
            key="outlier_col"
        )

        Q1 = df_used[selected_col_outliers].quantile(0.25)
        Q3 = df_used[selected_col_outliers].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_used[
            (df_used[selected_col_outliers] < lower_bound) |
            (df_used[selected_col_outliers] > upper_bound)
        ]

        st.write(f"Număr outliers: {len(outliers)}")
        st.write(f"Procent outliers: {(len(outliers) / len(df_used)) * 100:.2f}%")
        st.dataframe(outliers.head())

        if st.button("Elimină outliers"):
            df_no_outliers = df_used[
                (df_used[selected_col_outliers] >= lower_bound) &
                (df_used[selected_col_outliers] <= upper_bound)
            ]

            st.session_state["df_final"] = df_no_outliers
            st.success("Outliers eliminați!")
            st.write("Dimensiune nouă:", df_no_outliers.shape)
    else:
        st.write("Nu există coloane numerice pentru detectarea outliers.")

    if "df_final" in st.session_state:
        st.subheader("Date după curățare completă")
        st.dataframe(st.session_state["df_final"].head())

    st.subheader("Eliminare outliers pentru tot dataset-ul")

    if len(numeric_cols) > 0:
        if st.button("Elimină outliers din toate coloanele"):

            df_no_outliers_all = df_used.copy()

            for col in numeric_cols:
                Q1 = df_no_outliers_all[col].quantile(0.25)
                Q3 = df_no_outliers_all[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df_no_outliers_all = df_no_outliers_all[
                    (df_no_outliers_all[col] >= lower_bound) &
                    (df_no_outliers_all[col] <= upper_bound)
                    ]

            st.session_state["df_final_all"] = df_no_outliers_all

            st.success("Outliers eliminați din toate coloanele!")
            st.write("Dimensiune nouă:", df_no_outliers_all.shape)
    else:
        st.write("Nu există coloane numerice.")
else:
    st.warning("Încarcă mai întâi un dataset.")

st.markdown("<a id='groupby'></a>", unsafe_allow_html=True)
st.header("Analiză prin grupare și agregare")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Creare variabilă categorică din variabilă numerică")
    st.write("""
    Transformăm coloana numerică `price_usd` într-o variabilă categorică
    cu 3 clase: low / medium / high, pe baza cuantilelor.
    """)

    if "price_usd" in df.columns:
        q33 = df["price_usd"].quantile(0.33)
        q66 = df["price_usd"].quantile(0.66)


        def categorize_price(price):
            if price <= q33:
                return "low"
            elif price <= q66:
                return "medium"
            else:
                return "high"


        df["price_category"] = df["price_usd"].apply(categorize_price)
        st.session_state["df"] = df  # salvăm cu noua coloană

        st.write(f"Praguri: low ≤ {q33:.2f} | medium ≤ {q66:.2f} | high > {q66:.2f}")
        st.write("Distribuție clase:")
        st.write(df["price_category"].value_counts())

        cat_group = df.groupby("price_category")["prop_review_score"].agg(["mean", "count"]).reset_index()
        cat_group.columns = ["Categorie preț", "Review mediu", "Nr. înregistrări"]
        st.write("Analiză grupată pe categoria de preț:")
        st.dataframe(cat_group)
        st.bar_chart(cat_group.set_index("Categorie preț")["Review mediu"])
    else:
        st.warning("Coloana price_usd nu există.")

    st.subheader("Problema 1: Număr înregistrări per categorie")
    st.write("""
    Determinăm numărul de înregistrări pentru fiecare valoare distinctă
    a coloanei selectate.
    """)

    col = st.selectbox("Alege coloană pentru grupare", df.columns, key="p1")

    group_count = df.groupby(col).size().reset_index(name="count")
    if col == "prop_id":
        group_count["label"] = "Hotel " + group_count["prop_id"].astype(str)
    elif col == "srch_destination_id":
        group_count["label"] = "Zona " + group_count["srch_destination_id"].astype(str)
    else:
        group_count["label"] = group_count[col].astype(str)

    st.dataframe(group_count.head(20))
    st.bar_chart(group_count.set_index("label"))

    st.subheader("Problema 2: Preț mediu pe destinație")
    st.write("""
    Determinăm prețul mediu al hotelurilor pentru fiecare destinație.
    """)

    if "srch_destination_id" in df.columns and "price_usd" in df.columns:

        group_mean = df.groupby("srch_destination_id")["price_usd"].mean().reset_index()
        group_mean["destination_name"] = "Zona " + group_mean["srch_destination_id"].astype(str)

        st.dataframe(group_mean[["destination_name", "price_usd"]].head(20))
        st.bar_chart(group_mean.set_index("destination_name").head(20))

    else:
        st.warning("Coloanele necesare nu există.")

    st.subheader("Problema 3: Suma prețurilor pe destinație")
    st.write("""
    Calculăm suma prețurilor pentru fiecare destinație.
    """)

    if "srch_destination_id" in df.columns and "price_usd" in df.columns:

        group_sum = df.groupby("srch_destination_id")["price_usd"].sum().reset_index()
        group_sum["destination_name"] = "Zona " + group_sum["srch_destination_id"].astype(str)

        st.dataframe(group_sum[["destination_name", "price_usd"]].head(20))

    else:
        st.warning("Coloanele necesare nu există.")
# PROBLEMA 4
    st.subheader("Problema 4: Agregare multiplă pe destinație")
    st.write("""
    Pentru fiecare destinație determinăm:
    - media
    - minimul
    - maximul prețurilor
    """)

    if "srch_destination_id" in df.columns and "price_usd" in df.columns:

        group_agg = df.groupby("srch_destination_id")["price_usd"].agg(["mean", "min", "max"]).reset_index()
        group_agg["destination_name"] = "Zona " + group_agg["srch_destination_id"].astype(str)

        st.dataframe(group_agg[["destination_name", "mean", "min", "max"]].head(20))

    else:
        st.warning("Coloanele necesare nu există.")

# PROBLEMA 5
    st.subheader("Problema 5: Top hoteluri după performanță")
    st.write("""
    Determinăm performanța hotelurilor pe baza:
    - numărului de apariții
    - prețului mediu
    - prețului maxim
    """)

    hotel_stats = df.groupby("prop_id").agg({
        "price_usd": ["count", "mean", "max"]
    })

    hotel_stats.columns = ["nr_aparitii", "pret_mediu", "pret_maxim"]
    hotel_stats = hotel_stats.reset_index()
    hotel_stats["hotel_name"] = "Hotel " + hotel_stats["prop_id"].astype(str)

    hotel_stats_sorted = hotel_stats.sort_values(
        by=["nr_aparitii", "pret_mediu"],
        ascending=[False, False]
    )

    st.dataframe(hotel_stats_sorted[["hotel_name", "nr_aparitii", "pret_mediu", "pret_maxim"]].head(20))

    st.subheader("Problema 6: Analiză pe destinație și hotel")
    st.write("""
    Pentru fiecare combinație destinație-hotel determinăm:
    - numărul de apariții
    - prețul mediu
    """)

    multi_group = df.groupby(
        ["srch_destination_id", "prop_id"]
    ).agg({
        "price_usd": ["count", "mean"]
    })

    multi_group.columns = ["nr_aparitii", "pret_mediu"]
    multi_group = multi_group.reset_index()
    multi_group["hotel_name"] = "Hotel " + multi_group["prop_id"].astype(str)
    multi_group["destination_name"] = "Zona " + multi_group["srch_destination_id"].astype(str)

    st.dataframe(multi_group[["destination_name", "hotel_name", "nr_aparitii", "pret_mediu"]].head(20))

    # top
    top_dest = multi_group.sort_values(
        by="nr_aparitii",
        ascending=False
    )

    st.write("Top combinații:")
    st.dataframe(top_dest[["destination_name", "hotel_name", "nr_aparitii", "pret_mediu"]].head(10))

    st.subheader("Problema 7: Identificarea hotelurilor supraevaluate")
    st.write("""
    Determinăm hotelurile care au prețuri mai mari decât media destinației.
    """)

    if "srch_destination_id" in df.columns and "prop_id" in df.columns and "price_usd" in df.columns:
        # media pe destinație
        dest_mean = df.groupby("srch_destination_id")["price_usd"].mean().reset_index()
        dest_mean = dest_mean.rename(columns={"price_usd": "dest_mean_price"})
        # media pe hotel
        hotel_mean = df.groupby(["srch_destination_id", "prop_id"])["price_usd"].mean().reset_index()
        hotel_mean = hotel_mean.rename(columns={"price_usd": "hotel_mean_price"})

        merged = pd.merge(hotel_mean, dest_mean, on="srch_destination_id")

        merged["price_diff"] = merged["hotel_mean_price"] - merged["dest_mean_price"]

        overpriced = merged[merged["price_diff"] > 0]
        overpriced["hotel_name"] = "Hotel " + overpriced["prop_id"].astype(str)
        overpriced["destination_name"] = "Zona " + overpriced["srch_destination_id"].astype(str)

        overpriced = overpriced.sort_values(by="price_diff", ascending=False)

        st.write("Top hoteluri supraevaluate:")
        st.dataframe(overpriced[["destination_name", "hotel_name", "price_diff"]].head(20))

        # grafic
        st.subheader("Grafic diferență de preț")
        top10 = overpriced.head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = "Hotel " + top10["prop_id"].astype(str)
        ax.bar(labels, top10["price_diff"])
        ax.set_title("Top hoteluri supraevaluate")
        ax.set_xlabel("Hotel (prop_id)")
        ax.set_ylabel("Diferență preț")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

    else:
        st.warning("Coloanele necesare nu există.")

else:
    st.warning("Nu există date încă. Te rog încarcă fișierul CSV.")

st.markdown("<a id='prep'></a>", unsafe_allow_html=True)
st.header("Preprocesare - Codificare date")

if "df" in st.session_state:
    df_base = st.session_state.get("df_final", st.session_state.get("df_clean", st.session_state["df"]))

    # --------------------------- ENCODING ---------------------------
    st.subheader("Codificarea variabilelor categorice")
    st.write("""
    Datasetul Expedia nu conține coloane de tip text, dar are coloane numerice cu valori discrete
    (ex: număr stele, flag-uri 0/1) care pot fi tratate ca variabile categorice și encodate.
    """)

    COLS_TO_ENCODE = [
        "prop_starrating", "prop_brand_bool", "promotion_flag",
        "srch_saturday_night_bool", "random_bool", "booking_bool"
    ]
    cols_to_encode = [c for c in COLS_TO_ENCODE if c in df_base.columns]

    df_encoded = df_base.copy()

    st.write("Coloane selectate pentru Label Encoding:")
    st.write(cols_to_encode)

    if st.button("Aplică Label Encoding"):
        encoders = {}
        for col in cols_to_encode:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

        st.session_state["df_encoded"] = df_encoded
        st.success("Codificarea a fost aplicată.")
        st.dataframe(df_encoded[cols_to_encode].head())

        for col, le in encoders.items():
            mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
            st.write(f"Mapping `{col}`: {mapping}")

    if "df_encoded" in st.session_state:
        st.subheader("Dataset după encoding")
        st.dataframe(st.session_state["df_encoded"].head())

    st.subheader("Scalarea variabilelor numerice")

    df_scale_input = st.session_state.get("df_encoded", df_base).copy()
    numeric_cols = df_scale_input.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cols_to_exclude = ["prop_id", "srch_destination_id", "srch_id", "booking_bool", "click_bool"]
    numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]

    if len(numeric_cols) > 0:
        st.write("Coloane numerice folosite pentru scaling:")
        st.write(numeric_cols)

        if st.button("Aplică Standard Scaling"):
            scaler = StandardScaler()
            df_scaled = df_scale_input.copy()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

            st.session_state["df_scaled"] = df_scaled
            st.success("Scalarea a fost aplicată.")
            st.dataframe(df_scaled.head())

            st.write("Statistici după scalare (medie ≈ 0, std ≈ 1):")
            st.write(df_scaled[numeric_cols].describe())
    else:
        st.write("Nu există coloane numerice pentru scaling.")

    if "df_scaled" in st.session_state:
        st.subheader("Dataset final după preprocesare")
        st.dataframe(st.session_state["df_scaled"].head())
        st.write("Dimensiune dataset:", st.session_state["df_scaled"].shape)

    from sklearn.cluster import KMeans

    st.subheader("Clusterizare (KMeans)")

    if "df_scaled" in st.session_state:
        df_cluster = st.session_state["df_scaled"].copy()

        cluster_features_candidates = [
            "price_usd", "prop_starrating", "prop_review_score",
            "prop_location_score1", "prop_location_score2"
        ]
        cluster_features = [f for f in cluster_features_candidates if f in df_cluster.columns]

        st.write("Features folosite pentru clustering:", cluster_features)

        if len(cluster_features) > 0:
            k = st.slider("Alege numărul de clustere (k)", 2, 6, 3)

            if st.button("Aplică KMeans"):
                df_cluster_clean = df_cluster[cluster_features].dropna()
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                df_cluster.loc[df_cluster_clean.index, "cluster"] = kmeans.fit_predict(df_cluster_clean)

                st.session_state["df_cluster"] = df_cluster

                st.success("Clusterizare realizată!")
                st.dataframe(df_cluster[cluster_features + ["cluster"]].head(10))

                st.write("Număr elemente per cluster:")
                st.write(df_cluster["cluster"].value_counts().sort_index())
        else:
            st.warning("Coloanele necesare nu există.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    st.subheader("Regresie Logistică")
    st.write("""
    Scopul: prezicerea dacă o căutare va duce la o rezervare (`booking_bool`),
    pe baza caracteristicilor hotelului și ale căutării.
    """)

    if "df_scaled" in st.session_state:
        df_model = st.session_state["df_scaled"].copy()

        target = "booking_bool"
        logreg_features = [
            "price_usd", "prop_starrating", "prop_review_score",
            "prop_location_score1", "srch_length_of_stay",
            "srch_adults_count", "promotion_flag"
        ]
        logreg_features = [f for f in logreg_features if f in df_model.columns and f != target]

        df_model_clean = df_model[logreg_features + [target]].dropna()

        X = df_model_clean[logreg_features]
        y = df_model_clean[target]

        st.write(f"Features: {logreg_features}")
        st.write(f"Target: `{target}` | Distribuție: {y.value_counts().to_dict()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if st.button("Antrenează model Logistic Regression"):
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model antrenat! Acuratețe: {acc:.4f}")
            st.text(classification_report(y_test, y_pred))

    import statsmodels.api as sm

    st.subheader("Regresie multiplă (StatsModels)")
    st.write("""
    Scopul: modelarea prețului hotelului (`price_usd`) în funcție de
    caracteristici ale proprietății.
    """)

    if "df_scaled" in st.session_state:
        df_reg = st.session_state["df_scaled"].copy()

        reg_features = [
            "prop_location_score1",
            "prop_location_score2",
            "prop_review_score",
            "prop_starrating"
        ]
        target = "price_usd"

        reg_features = [col for col in reg_features if col in df_reg.columns]

        if len(reg_features) > 0 and target in df_reg.columns:
            df_reg_clean = df_reg[reg_features + [target]].dropna()

            X = df_reg_clean[reg_features]
            y = df_reg_clean[target]
            X = sm.add_constant(X)

            if st.button("Rulează regresia multiplă"):
                model = sm.OLS(y, X).fit()
                st.success("Modelul a fost antrenat!")
                st.text(model.summary())
        else:
            st.warning("Nu există suficiente coloane pentru regresie.")

else:
    st.warning("Încarcă mai întâi un dataset.")