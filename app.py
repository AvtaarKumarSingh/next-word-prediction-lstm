import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="NextWord",
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
.dataset-subtitle {
    color: #b7bfd8;
    font-size: 0.92rem;
    line-height: 1.6;
}
.instruction {
    background: rgba(167,139,250,0.09);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
    padding: 14px 16px;
    color: #d8d7f5;
    line-height: 1.7;
    margin: 12px 0 20px;
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
        "`lstm.keras` aur `tokenizer.pkl` app.py ke folder mein hain."
    )
    st.stop()


@st.cache_data
def load_dataset():
    dataset_path = Path(__file__).parent / "data.csv"
    return pd.read_csv(dataset_path)


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


def get_words(text):
    import re
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", str(text).lower())


def find_matching_sentences(text, dataset):
    input_words = get_words(text)

    if not input_words or "data" not in dataset.columns:
        return []

    matches = []

    for sentence in dataset["data"].dropna().astype(str):
        sentence_words = get_words(sentence)

        if len(sentence_words) >= len(input_words):
            if sentence_words[:len(input_words)] == input_words:
                matches.append(sentence)

    return matches


def get_sentence_continuation(sentence, text, count):
    sentence_words = sentence.split()
    input_count = len(text.strip().split())
    remaining_words = sentence_words[input_count:]
    predicted_words = remaining_words[:count]
    result = text.strip()

    if predicted_words:
        result += " " + " ".join(predicted_words)

    sentence_completed = len(remaining_words) <= count

    return result, len(predicted_words), sentence_completed


left_col, right_col = st.columns([7, 3], gap="large")


with left_col:
    st.markdown("""
    <div class="hero">
        <h1>✨ NextWord </h1>
        <p>Your thought. Our prediction. Complete the sentence.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✍️ Start Your Sentence")

    st.markdown("""
    <div class="instruction">
        💡 <b>Ready to complete your thought?</b><br>
        Start with the opening words of a sentence for the best prediction.
        The model will help you discover what comes next!
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area(
        "Start typing",
        placeholder="Example: The password reset may",
        height=110,
        label_visibility="collapsed"
    )

    word_count = len(text.split()) if text.strip() else 0

    st.caption(
        f"📝 Words entered: {word_count} · Minimum required: 2"
    )

    number_of_words = st.slider(
        "How many words should be predicted?",
        min_value=1,
        max_value=10,
        value=3
    )

    if st.button("✨ Predict Next Words"):
        if word_count < 2:
            st.warning(
                "Please enter at least 2 words to get a prediction."
            )

        elif not tokenizer.texts_to_sequences([text.strip()])[0]:
            st.warning(
                "Please enter words that the model recognizes."
            )

        else:
            with st.spinner("Generating your prediction..."):
                try:
                    df = load_dataset()
                    matches = find_matching_sentences(text, df)
                except Exception:
                    matches = []

                if matches:
                    st.markdown("### 💡 Sentence Completion")

                    shown = 0

                    for sentence in matches[:5]:
                        result, added_count, completed = get_sentence_continuation(
                            sentence,
                            text,
                            number_of_words
                        )

                        if added_count == 0:
                            continue

                        shown += 1

                        st.markdown(
                            f'<div class="result">{result}</div>',
                            unsafe_allow_html=True
                        )

                        if completed:
                            st.caption("Sentence completed.")
                        else:
                            st.caption(
                                f"Added {added_count} word(s) from the dataset."
                            )

                    if shown == 0:
                        st.info(
                            "This sentence has no additional words in the dataset."
                        )

                    if len(matches) > 5:
                        st.caption(
                            "Showing 5 matching sentences. Try adding more opening words for a more specific result."
                        )

                else:
                    prediction = predict_next_words(
                        model,
                        tokenizer,
                        text,
                        number_of_words
                    )

                    st.markdown("### 💡 LSTM Prediction")

                    st.markdown(
                        f'<div class="result">{prediction}</div>',
                        unsafe_allow_html=True
                    )

                    st.caption(
                        "No matching sentence was found in the dataset, "
                        "so the LSTM model generated the continuation."
                    )


with right_col:
    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 2rem;">📚 Dataset Explorer</h1>
        <p>Explore the text behind the predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="dataset-subtitle">'
        'Explore the training sentences and understand the '
        'text patterns used by the model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    try:
        df = load_dataset()

        stat1, stat2 = st.columns(2)

        with stat1:
            st.metric("📄 Total Rows", f"{len(df):,}")

        with stat2:
            st.metric("📊 Total Columns", len(df.columns))

        st.write("")

        st.caption(f"Showing all {len(df):,} rows")

        st.dataframe(
            df,
            height=250,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "↕️ Scroll inside the table to explore the complete dataset."
        )

        st.download_button(
            label="⬇️ Download Full Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="training_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    except FileNotFoundError:
        st.error(
            "`data.csv` nahi mili. Is file ko app.py ke same "
            "folder mein rakho aur GitHub par push karo."
        )

    except Exception as e:
        st.error(f"Dataset load nahi ho paaya: {e}")


st.markdown("""
<div style="text-align:center; color:#929bb8; padding:30px 0 10px;">
    Built with Python · Streamlit · TensorFlow
</div>
""", unsafe_allow_html=True)
