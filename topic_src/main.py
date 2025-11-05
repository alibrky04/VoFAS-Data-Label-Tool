import json
import pandas as pd
import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk
import os

# ---- SETTINGS ----
DATA_FILE = "datasets/dataset_preprocessed.json"
NUM_TOPICS = 10
TOP_WORDS = 10

OUTPUT_DIR = "topics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- PREPARE STOPWORDS ----
nltk.download("stopwords")

tr_stopwords = set(stopwords.words("turkish"))
en_stopwords = set(stopwords.words("english"))

extra_tr = {"havaalanı","havalimanı","alanı","hava","limanı","bir","çok","şey","var","daha","bile","olan"}
extra_en = {"airport","airline","flight","plane","terminal","one","get","got","us","like"}

stopwords_map = {
    "tr": tr_stopwords.union(extra_tr),
    "en": en_stopwords.union(extra_en),
}

def clean_text(text, lang="tr"):
    if not isinstance(text, str):
        return ""
    stop_words = stopwords_map.get(lang, set())
    text = text.lower()
    
    # language-specific character filtering
    if lang == "tr":
        text = re.sub(r"[^a-zçğıöşü\s]", "", text)
    else:
        text = re.sub(r"[^a-z\s]", "", text)
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def run_lda(df_lang, lang):
    if df_lang.empty:
        print(f"\n⚠️ No data for language: {lang}, skipping LDA")
        return

    texts = df_lang["clean_text"].tolist()
    text_tokens = [t.split() for t in texts]

    dictionary = corpora.Dictionary(text_tokens)
    corpus = [dictionary.doc2bow(text) for text in text_tokens]

    print(f"\n🔧 Training LDA for '{lang}' on {len(text_tokens)} reviews...")

    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=10,
        random_state=42
    )

    topics = []
    for idx, topic in lda_model.print_topics(num_topics=NUM_TOPICS, num_words=TOP_WORDS):
        topics.append({"topic_id": idx, "keywords": topic})

    out_csv = f"{OUTPUT_DIR}/{lang}_topics.csv"
    out_json = f"{OUTPUT_DIR}/{lang}_topics.json"

    pd.DataFrame(topics).to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {lang.upper()} topics to: {out_csv}, {out_json}\n")

    for t in topics:
        print(f"[{lang}] Topic {t['topic_id']}: {t['keywords']}")


# ---- LOAD DATA ----
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ---- CLEAN TEXT PER LANGUAGE ----
df["clean_text"] = df.apply(lambda r: clean_text(r["review_text"], r.get("language", "tr")), axis=1)

# ---- SPLIT ----
df_tr = df[df["language"] == "tr"].copy()
df_en = df[df["language"] == "en"].copy()

print(f"📊 Turkish reviews: {len(df_tr)}")
print(f"📊 English reviews: {len(df_en)}")

# ---- RUN TOPIC MODELS ----
run_lda(df_tr, "tr")
run_lda(df_en, "en")