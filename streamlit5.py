import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@st.cache_data
def load_data():
    df = pd.read_csv("hasil_preprocessing_lengkap2.csv")
    df = df[df['Rating'] != 3]
    df['Sentimen'] = df['Rating'].apply(lambda x: 'Positif' if x >= 4 else 'Negatif')
    df = df.dropna(subset=['stemming'])
    return df

data = load_data()

@st.cache_resource
def prepare_models(df):
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_all = tfidf.fit_transform(df['stemming'])
    y_all = df['Sentimen'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, df.index, test_size=0.2, random_state=42, stratify=y_all
    )

    train_texts = df.loc[idx_train]

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    return tfidf, nb, lr, X_train, y_train, train_texts

tfidf, nb_model, lr_model, X_train, y_train, train_texts = prepare_models(data)

def preprocess_text(text):
    text = text.lower()
    return ''.join(c for c in text if c.isalnum() or c.isspace())

def predict_sentiment(text, model):
    clean = preprocess_text(text)
    tfidf_input = tfidf.transform([clean])
    label = model.predict(tfidf_input)[0]
    prob = model.predict_proba(tfidf_input)[0]
    return label, prob

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='Positif')
    rec = recall_score(y_true, y_pred, pos_label='Positif')
    f1 = f1_score(y_true, y_pred, pos_label='Positif')
    cm = confusion_matrix(y_true, y_pred, labels=['Positif', 'Negatif'])
    return acc, prec, rec, f1, cm, y_pred

st.set_page_config("Analisis Sentimen DANA", layout="wide")
menu = st.sidebar.radio("ANALISIS SENTIMEN APLIKASI DANA", ["🏠 Dashboard", "📊 Confusion Matrix", "💬 Wordcloud", "📈 Perbandingan Metrik", "🔍 Prediksi Ulasan Baru"])

if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Ulasan Aplikasi DANA")
    st.write("Jumlah data setelah preprocessing:", len(data))
    count_sentiment = data['Sentimen'].value_counts()
    st.subheader("Distribusi Sentimen")
    st.bar_chart(count_sentiment)
    st.subheader("Contoh Data")
    st.dataframe(data[['stemming', 'Rating', 'Sentimen']].sample(10))

elif menu == "📊 Confusion Matrix":
    st.title("📊 Confusion Matrix (Data Latih)")
    acc_nb, _, _, _, cm_nb, _ = evaluate_model(nb_model, X_train, y_train)
    acc_lr, _, _, _, cm_lr, _ = evaluate_model(lr_model, X_train, y_train)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naive Bayes")
        st.write(f"Akurasi: {acc_nb:.2f}")
        fig, ax = plt.subplots()
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', xticklabels=['Positif','Negatif'], yticklabels=['Positif','Negatif'], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
    with col2:
        st.subheader("Logistic Regression")
        st.write(f"Akurasi: {acc_lr:.2f}")
        fig, ax = plt.subplots()
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', xticklabels=['Positif','Negatif'], yticklabels=['Positif','Negatif'], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

elif menu == "💬 Wordcloud":
    st.title("💬 Wordcloud Prediksi Model (Data Latih)")
    _, _, _, _, _, nb_pred = evaluate_model(nb_model, X_train, y_train)
    _, _, _, _, _, lr_pred = evaluate_model(lr_model, X_train, y_train)

    def plot_wordcloud(predictions, label, model_name, color):
        filtered = train_texts[predictions == label]
        text = ' '.join(filtered['stemming'])
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white', colormap=color).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write(f"Tidak ada ulasan untuk label {label}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naive Bayes - Positif")
        plot_wordcloud(nb_pred, 'Positif', "Naive Bayes", 'Greens')
        st.subheader("Naive Bayes - Negatif")
        plot_wordcloud(nb_pred, 'Negatif', "Naive Bayes", 'Reds')
    with col2:
        st.subheader("Logistic Regression - Positif")
        plot_wordcloud(lr_pred, 'Positif', "Logistic Regression", 'Greens')
        st.subheader("Logistic Regression - Negatif")
        plot_wordcloud(lr_pred, 'Negatif', "Logistic Regression", 'Reds')

elif menu == "📈 Perbandingan Metrik":
    st.title("📈 Perbandingan Metrik Model (Data Latih)")
    acc_nb, prec_nb, rec_nb, f1_nb, _, _ = evaluate_model(nb_model, X_train, y_train)
    acc_lr, prec_lr, rec_lr, f1_lr, _, _ = evaluate_model(lr_model, X_train, y_train)
    metrics = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    nb_scores = [acc_nb, prec_nb, rec_nb, f1_nb]
    lr_scores = [acc_lr, prec_lr, rec_lr, f1_lr]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.barh(x - width/2, nb_scores, height=0.35, label="Naive Bayes", color="#2e8b57")
    bars2 = ax.barh(x + width/2, lr_scores, height=0.35, label="Logistic Regression", color="#32cd32")
    ax.set_yticks(x)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Skor")
    ax.set_title("Perbandingan Metrik")
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
    for bar in bars2:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
    st.pyplot(fig)

elif menu == "🔍 Prediksi Ulasan Baru":
    st.title("🔍 Prediksi Ulasan Baru dari File")

    uploaded_file = st.file_uploader("Unggah file CSV atau Excel yang berisi kolom ulasan", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file)

            st.write("📋 Kolom ditemukan:", df_new.columns.tolist())
            st.dataframe(df_new.head())

            # Jika ada kolom rating, buat label otomatis
            if 'rating' in df_new.columns:
                df_new = df_new[df_new['rating'] != 3]  # Hilangkan rating netral
                df_new['label'] = df_new['rating'].apply(lambda x: 'Positif' if x > 3 else 'Negatif')

            if 'ulasan' not in df_new.columns:
                st.error("❌ Kolom 'ulasan' tidak ditemukan. Harap pastikan nama kolom tepat.")
                st.stop()

            df_new = df_new.dropna(subset=['ulasan'])
            if df_new['ulasan'].dropna().empty:
                st.error("❌ Kolom 'ulasan' kosong. Harap isi dengan data yang valid.")
                st.stop()

            # Preprocessing
            df_new['clean'] = df_new['ulasan'].apply(preprocess_text)
            tfidf_new = tfidf.transform(df_new['clean'])

            # Prediksi
            df_new['Prediksi_NB'] = nb_model.predict(tfidf_new)
            df_new['Prediksi_LR'] = lr_model.predict(tfidf_new)

            # Grafik batang distribusi prediksi
            st.subheader("📊 Grafik Jumlah Prediksi Sentimen")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Naive Bayes**")
                nb_counts = df_new['Prediksi_NB'].value_counts()
                fig_nb = px.bar(
                    x=nb_counts.index,
                    y=nb_counts.values,
                    labels={'x': 'Sentimen', 'y': 'Jumlah'},
                    color=nb_counts.index,
                    color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'},
                    title='Distribusi Sentimen - Naive Bayes'
                )
                st.plotly_chart(fig_nb, use_container_width=True)

            with col2:
                st.markdown("**Logistic Regression**")
                lr_counts = df_new['Prediksi_LR'].value_counts()
                fig_lr = px.bar(
                    x=lr_counts.index,
                    y=lr_counts.values,
                    labels={'x': 'Sentimen', 'y': 'Jumlah'},
                    color=lr_counts.index,
                    color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'},
                    title='Distribusi Sentimen - Logistic Regression'
                )
                st.plotly_chart(fig_lr, use_container_width=True)

            # Confusion Matrix jika ada label asli
            if 'label' in df_new.columns:
                st.subheader("📊 Confusion Matrix")

                cm_nb = confusion_matrix(df_new['label'], df_new['Prediksi_NB'], labels=['Positif','Negatif'])
                cm_lr = confusion_matrix(df_new['label'], df_new['Prediksi_LR'], labels=['Positif','Negatif'])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Naive Bayes**")
                    fig1, ax1 = plt.subplots()
                    sns.heatmap(cm_nb, annot=True, fmt='.0f', cmap='Blues',
                                xticklabels=['Positif','Negatif'], yticklabels=['Positif','Negatif'], ax=ax1)
                    ax1.set_xlabel("Prediksi")
                    ax1.set_ylabel("Aktual")
                    st.pyplot(fig1)

                with col2:
                    st.markdown("**Logistic Regression**")
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(cm_lr, annot=True, fmt='.0f', cmap='YlOrBr',
                                xticklabels=['Positif','Negatif'], yticklabels=['Positif','Negatif'], ax=ax2)
                    ax2.set_xlabel("Prediksi")
                    ax2.set_ylabel("Aktual")
                    st.pyplot(fig2)

                # Metrik
                acc_nb = accuracy_score(df_new['label'], df_new['Prediksi_NB'])
                acc_lr = accuracy_score(df_new['label'], df_new['Prediksi_LR'])
                prec_nb = precision_score(df_new['label'], df_new['Prediksi_NB'], pos_label='Positif')
                prec_lr = precision_score(df_new['label'], df_new['Prediksi_LR'], pos_label='Positif')
                rec_nb = recall_score(df_new['label'], df_new['Prediksi_NB'], pos_label='Positif')
                rec_lr = recall_score(df_new['label'], df_new['Prediksi_LR'], pos_label='Positif')
                f1_nb = f1_score(df_new['label'], df_new['Prediksi_NB'], pos_label='Positif')
                f1_lr = f1_score(df_new['label'], df_new['Prediksi_LR'], pos_label='Positif')

                df_metrik = pd.DataFrame({
                    'Model': ['Naive Bayes', 'Logistic Regression'],
                    'Akurasi': [round(acc_nb, 2), round(acc_lr, 2)],
                    'Presisi': [round(prec_nb, 2), round(prec_lr, 2)],
                    'Recall': [round(rec_nb, 2), round(rec_lr, 2)],
                    'F1-Score': [round(f1_nb, 2), round(f1_lr, 2)]
                })

                st.subheader("📈 Perbandingan Metrik (Bar Chart)")
                fig_bar = px.bar(
                    df_metrik.melt(id_vars='Model'),
                    x='value', y='variable', color='Model', orientation='h',
                    barmode='group',
                    labels={'value': 'Skor', 'variable': 'Metrik'},
                    color_discrete_map={'Naive Bayes': '#2980b9', 'Logistic Regression': '#f1c40f'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)


        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")
    else:
        st.info("Silakan unggah file CSV/XLSX terlebih dahulu.")