
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="NextWord AI",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #101323, #17152b, #101b2b);
    color: #f5f5ff;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}
.hero {
    text-align: center;
    padding: 25px 10px;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #b7bfd8;
    font-size: 1.05rem;
}
.result {
    background: rgba(103,232,249,0.08);
    border: 1px solid rgba(103,232,249,0.35);
    border-radius: 16px;
    padding: 22px;
    font-size: 1.2rem;
    line-height: 1.8;
    overflow-wrap: anywhere;
}
.stButton > button {
    width: 100%;
    min-height: 3rem;
    border: none;
    border-radius: 12px;
    color: #101323;
    font-weight: 750;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
}
.stButton > button:hover {
    border: none;
    color: #101323;
    filter: brightness(1.08);
}
.dataset-card {
    background: rgba(103,232,249,0.045);
    border: 1px solid rgba(167,139,250,0.28);
    border-radius: 16px;
    padding: 18px;
}
.dataset-subtitle {
    color: #b7bfd8;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = load_model("lstm.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


try:
    model, tokenizer = load_assets()
except Exception:
    st.error(
        "Model load nahi hua. Check karo ki "
        "`lstm.keras` aur `tokenizer.pkl` "
        "app.py ke folder mein hain."
    )
    st.stop()




DATASET_PATH = Path(__file__).parent / "data.csv"


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


# ---------------- PREDICTION LOGIC ----------------

def predict_next_words(model, tokenizer, text, count):
    index_to_word = {
        index: word
        for word, index in tokenizer.word_index.items()
    }

    result = text.strip()
    max_len = model.input_shape[1]
    output_size = model.output_shape[-1]

    for _ in range(count):
        sequence = tokenizer.texts_to_sequences([result])[0]

        if not sequence:
            break

        padded = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="pre",
            truncating="pre"
        )

        probabilities = model.predict(padded, verbose=0)[0].copy()

        valid_indices = [
            index for index in index_to_word
            if 0 < index < output_size
        ]

        if not valid_indices:
            break

        allowed = np.full(output_size, -np.inf)
        allowed[valid_indices] = probabilities[valid_indices]

        next_index = int(np.argmax(allowed))
        next_word = index_to_word.get(next_index)

        if not next_word:
            break

        result += " " + next_word

    return result


# ---------------- TWO-COLUMN LAYOUT ----------------

left_col, right_col = st.columns(
    [1, 1.05],
    gap="large"
)




with left_col:

    st.markdown("""
    <div class="hero">
        <h1>✨ NextWord </h1>
        <p>Your thought. Our prediction. Complete the sentence.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✍️ Enter your sentence")

    text = st.text_area(
        "Start typing",
        placeholder="Example: The password reset may",
        height=110,
        label_visibility="collapsed"
    )

    word_count = len(text.split()) if text.strip() else 0

    st.caption(
        f"📝 Words entered: {word_count} · "
        "Minimum required: 2"
    )

    number_of_words = st.slider(
        "How many words should AI predict?",
        min_value=1,
        max_value=10,
        value=3
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✨ Predict Next Words"):
        if word_count < 2:
            st.warning(
                "Bhai, prediction ke liye kam se kam "
                "2 words enter kar! 😊"
            )

        elif not tokenizer.texts_to_sequences([text.strip()])[0]:
            st.warning(
                "Input mein tokenizer ke pehchane hue "
                "words hone chahiye."
            )

        else:
            with st.spinner("AI is thinking..."):
                prediction = predict_next_words(
                    model,
                    tokenizer,
                    text,
                    number_of_words
                )

            st.markdown("### 💡 Prediction")

            st.markdown(
                f'<div class="result">{prediction}</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"Requested: {number_of_words} next words · "
                "Actual output may be shorter if no valid "
                "prediction is available."
            )

            st.button(
                "📋 Copy tip: Select the result text to copy it",
                disabled=True
            )


# ==================================================
# RIGHT SIDE: DATASET EXPLORER
# ==================================================

with right_col:

    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 2rem;">📚 Dataset Explorer</h1>
        <p>Explore the text behind the predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="dataset-subtitle">'
        'Browse the training data, search sentences, '
        'and explore the patterns learned by the model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    try:
        df = load_dataset()

        # Dataset statistics
        stat1, stat2 = st.columns(2)

        with stat1:
            st.metric("📄 Total Rows", f"{len(df):,}")

        with stat2:
            st.metric("📊 Total Columns", len(df.columns))

        st.write("")

        # Search dataset
        search_text = st.text_input(
            "🔎 Search the dataset",
            placeholder="Type a word or sentence...",
            key="dataset_search"
        )

        filtered_df = df

        if search_text.strip():
            mask = df.astype(str).apply(
                lambda column: column.str.contains(
                    search_text.strip(),
                    case=False,
                    regex=False,
                    na=False
                )
            ).any(axis=1)

            filtered_df = df[mask]

        st.caption(
            f"Showing {len(filtered_df):,} of "
            f"{len(df):,} rows"
        )

        # Fixed height: initially shows around 5-6 rows.
        # Scroll inside the table to explore the remaining rows.
        st.dataframe(
            filtered_df,
            height=250,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "↕️ Scroll inside the table to explore more rows. "
            "Click column headers to sort."
        )

        # Download complete dataset, not just search results
        st.download_button(
            label="⬇️ Download Full Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="training_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Sample text for prediction
        st.markdown("### 💡 Try a Dataset Example")

        if len(df.columns) > 0:
            text_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()

            if not text_columns:
                text_columns = df.columns.tolist()

            selected_column = st.selectbox(
                "Choose a text column",
                text_columns,
                key="example_column"
            )

            sentences = (
                df[selected_column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            sentences = sentences[sentences.str.len() > 0]

            if not sentences.empty:
                # Select up to five actual dataset examples
                examples = sentences.drop_duplicates().head(5)

                for i, sentence in enumerate(examples, start=1):
                    st.code(
                        f"{i}. {sentence}",
                        language=None
                    )

                st.caption(
                    "Copy an example into the predictor on the left. "
                    "These are dataset examples, not guaranteed "
                    "correct predictions."
                )

    except FileNotFoundError:
        st.error(
            "`data.csv` nahi mili. Is file ko `app.py` ke "
            "same folder mein rakho aur GitHub par push karo."
        )

    except Exception as e:
        st.error(f"Dataset load nahi ho paaya: {e}")



st.markdown("""
<div style="text-align:center; color:#929bb8; padding:30px 0 10px;">
    Built with Python · Streamlit · TensorFlow
</div>
""", unsafe_allow_html=True)
