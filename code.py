"""
UC03 - Social Media Sentiment Analysis AI Agent
FREE / NO-API VERSION

Run locally:
    pip install -r requirements.txt
    streamlit run code.py

Dataset expected in the same folder:
    Synthetic Data - Social Media Tweets.xlsx

This version uses a lightweight, explainable NLP sentiment engine built with
Python standard-library logic. It requires no OpenAI API key, paid API credits,
external model download, vector database, or generated database files.
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


APP_TITLE = "UC03 - Social Media Sentiment Analysis Agent"
DATASET_NAME = "Synthetic Data - Social Media Tweets.xlsx"

# Explainable sentiment vocabulary. The engine combines word polarity,
# negation, intensifiers, contrast, and common social-media expressions.
POSITIVE_WORDS = {
    "amazing", "awesome", "best", "better", "brilliant", "clean", "enjoy",
    "enjoyed", "excellent", "fast", "faster", "favorite", "good", "great",
    "happy", "helpful", "impressive", "improved", "improvement", "love",
    "loved", "nice", "perfect", "positive", "recommend", "smooth", "success",
    "successful", "thank", "thanks", "useful", "win", "wonderful", "works",
    "working", "easy", "excited", "pleased", "satisfied", "reliable",
    "stable", "fixed", "fix", "upgrade", "upgrade", "👍", "😀", "😃", "😄",
    "😁", "😊", "😍", "🎉", "❤", "❤️", "🔥", "✨", "💯", "+1",
}

NEGATIVE_WORDS = {
    "bad", "broken", "bug", "bugs", "buggy", "crash", "crashes", "crashing",
    "crashed", "complaint", "confusing", "delay", "delayed", "difficult",
    "disappointed", "disappointing", "disaster", "fail", "failed", "failure",
    "frustrated", "frustrating", "hate", "hated", "issue", "issues", "lag",
    "laggy", "late", "negative", "poor", "problem", "problems", "slow",
    "slower", "terrible", "unhappy", "useless", "worst", "wrong", "error",
    "errors", "annoying", "annoyed", "awful", "pain", "painful", "refund",
    "refunds", "downtime", "unreliable", "unstable", "😞", "😔", "😟", "😠",
    "😡", "🤬", "😢", "😭", "💔", "👎", "❌",
}

NEGATIONS = {
    "not", "no", "never", "neither", "nor", "hardly", "barely", "isn't",
    "wasn't", "weren't", "don't", "doesn't", "didn't", "can't", "cannot",
    "couldn't", "won't", "wouldn't", "shouldn't",
}

INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3,
    "super": 1.5, "absolutely": 1.8, "totally": 1.5, "incredibly": 1.7,
}

CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "surprisingly"}

PHRASE_SCORES = {
    "can't believe how good": 3.0,
    "couldn't be happier": 3.0,
    "works surprisingly well": 3.0,
    "surprisingly good": 2.5,
    "love this": 2.5,
    "highly recommend": 2.5,
    "works great": 2.5,
    "works well": 2.0,
    "very good": 2.0,
    "very bad": -2.0,
    "keeps crashing": -3.0,
    "constant crashing": -3.0,
    "doesn't work": -2.5,
    "not working": -2.5,
    "not good": -2.0,
    "not bad": 1.0,
    "no problem": 1.5,
    "no issues": 1.5,
    "terrible experience": -3.0,
    "waste of time": -3.0,
}


def locate_dataset() -> Path:
    candidates = [
        Path(__file__).resolve().parent / DATASET_NAME,
        Path.cwd() / DATASET_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{DATASET_NAME}'. Put the supplied Excel file in the same folder as code.py."
    )


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"Tweet_ID", "Tweet_Text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required column(s): " + ", ".join(sorted(missing))
        )
    df = df[["Tweet_ID", "Tweet_Text"]].copy()
    df["Tweet_ID"] = df["Tweet_ID"].astype(str).str.strip()
    df["Tweet_Text"] = df["Tweet_Text"].fillna("").astype(str).str.strip()
    return df[df["Tweet_Text"] != ""].reset_index(drop=True)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[+-]?\d+(?:\.\d+)?|[😀-🙏❤-🫶]", text.lower())


def retrieve_relevant_tweets(query: str, df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    query_terms = set(tokenize(query))
    query_lower = query.lower().strip()
    scored = []
    for _, row in df.iterrows():
        tweet_text = row["Tweet_Text"]
        tweet_terms = set(tokenize(tweet_text))
        overlap = len(query_terms & tweet_terms)
        phrase_bonus = 4 if query_lower and query_lower in tweet_text.lower() else 0
        scored.append({
            "Tweet_ID": row["Tweet_ID"],
            "Tweet_Text": tweet_text,
            "_score": overlap + phrase_bonus,
        })
    scored.sort(key=lambda item: item["_score"], reverse=True)
    return pd.DataFrame(scored[:top_k])


def format_tweets(df: pd.DataFrame) -> str:
    if df.empty:
        return "No dataset tweets were retrieved."
    return "\n".join(
        f"[{row.Tweet_ID}] {row.Tweet_Text}" for row in df.itertuples(index=False)
    )


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def sentiment_score(text: str) -> Tuple[float, List[str], List[str]]:
    """Return score, positive evidence, negative evidence."""
    text_lower = clean_text(text)
    tokens = tokenize(text_lower)
    score = 0.0
    positive_hits: List[str] = []
    negative_hits: List[str] = []

    for phrase, value in PHRASE_SCORES.items():
        if phrase in text_lower:
            score += value
            if value > 0:
                positive_hits.append(phrase)
            else:
                negative_hits.append(phrase)

    # Handle word-level polarity, local negation, and intensifiers.
    for i, token in enumerate(tokens):
        value = 0.0
        if token in POSITIVE_WORDS:
            value = 1.0
        elif token in NEGATIVE_WORDS:
            value = -1.0
        if value == 0:
            continue

        previous = tokens[max(0, i - 3):i]
        negated = any(word in NEGATIONS for word in previous)
        multiplier = 1.0
        if previous:
            multiplier = INTENSIFIERS.get(previous[-1], 1.0)

        if negated:
            value *= -0.8

        value *= multiplier
        score += value
        evidence = token
        if value > 0:
            positive_hits.append(evidence)
        else:
            negative_hits.append(evidence)

    # Content after contrast words usually carries more weight because it
    # expresses the final or corrected opinion.
    contrast_positions = [tokens.index(word) for word in CONTRAST_WORDS if word in tokens]
    if contrast_positions:
        pos = max(contrast_positions)
        after = " ".join(tokens[pos + 1:])
        after_score, _, _ = sentiment_score_simple(after)
        before = " ".join(tokens[:pos])
        before_score, _, _ = sentiment_score_simple(before)
        score = before_score * 0.5 + after_score * 1.25

    return score, positive_hits, negative_hits


def sentiment_score_simple(text: str) -> Tuple[float, List[str], List[str]]:
    """Non-recursive scoring helper used for contrast handling."""
    tokens = tokenize(text)
    score = 0.0
    pos_hits: List[str] = []
    neg_hits: List[str] = []
    for i, token in enumerate(tokens):
        value = 1.0 if token in POSITIVE_WORDS else -1.0 if token in NEGATIVE_WORDS else 0.0
        if value == 0:
            continue
        previous = tokens[max(0, i - 3):i]
        if any(word in NEGATIONS for word in previous):
            value *= -0.8
        if previous:
            value *= INTENSIFIERS.get(previous[-1], 1.0)
        score += value
        (pos_hits if value > 0 else neg_hits).append(token)
    return score, pos_hits, neg_hits


def classify_sentiment(text: str) -> Dict[str, object]:
    score, pos_hits, neg_hits = sentiment_score(text)
    if score >= 0.8:
        label = "Positive"
    elif score <= -0.8:
        label = "Negative"
    else:
        label = "Neutral"

    magnitude = abs(score)
    if magnitude >= 3:
        confidence = "High"
    elif magnitude >= 1.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    if label == "Positive":
        evidence = ", ".join(dict.fromkeys(pos_hits)) or "positive wording"
        reason = f"The overall wording expresses a favorable opinion, supported by: {evidence}."
    elif label == "Negative":
        evidence = ", ".join(dict.fromkeys(neg_hits)) or "negative wording"
        reason = f"The overall wording expresses an unfavorable opinion, supported by: {evidence}."
    else:
        if pos_hits and neg_hits:
            reason = "Positive and negative signals are mixed or balanced, so the overall sentiment is Neutral."
        elif not pos_hits and not neg_hits:
            reason = "The text is mainly factual or does not contain a strong opinion signal."
        else:
            reason = "The available sentiment signals are too weak to support a clear Positive or Negative label."

    return {
        "sentiment": label,
        "confidence": confidence,
        "score": round(score, 2),
        "reason": reason,
    }


def analyze_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for _, row in df.iterrows():
        result = classify_sentiment(row["Tweet_Text"])
        rows.append({
            "Tweet_ID": row["Tweet_ID"],
            "Tweet_Text": row["Tweet_Text"],
            "Sentiment": result["sentiment"],
            "Confidence": result["confidence"],
            "Score": result["score"],
        })
    result_df = pd.DataFrame(rows)
    counts = result_df["Sentiment"].value_counts().to_dict()
    total = len(result_df)
    distribution = {
        label: round((counts.get(label, 0) / total) * 100, 1) if total else 0.0
        for label in ["Positive", "Negative", "Neutral"]
    }
    return result_df, distribution


def is_overall_question(question: str) -> bool:
    q = question.lower()
    terms = [
        "overall sentiment", "overall analysis", "sentiment distribution",
        "all tweets", "entire dataset", "whole dataset", "dataset sentiment",
        "sentiment of the dataset", "analyze the dataset", "all of the tweets",
    ]
    return any(term in q for term in terms)


def is_dataset_search_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in ["find", "which tweets", "show tweets", "tweets about", "dataset"])


def answer_question(question: str, df: pd.DataFrame, history: List[Dict[str, str]]) -> str:
    q = question.strip()

    if is_overall_question(q):
        analyzed, distribution = analyze_dataset(df)
        counts = analyzed["Sentiment"].value_counts().to_dict()
        overall = max(
            ["Positive", "Negative", "Neutral"],
            key=lambda label: (counts.get(label, 0), distribution[label]),
        )
        return (
            f"### Overall Dataset Sentiment\n\n"
            f"**Total Tweets:** {len(analyzed)}\n\n"
            f"**Positive:** {counts.get('Positive', 0)} ({distribution['Positive']}%)  \n"
            f"**Negative:** {counts.get('Negative', 0)} ({distribution['Negative']}%)  \n"
            f"**Neutral:** {counts.get('Neutral', 0)} ({distribution['Neutral']}%)  \n\n"
            f"**Overall Sentiment:** {overall}\n\n"
            "**Key Insight:** The distribution is calculated directly from the supplied "
            "100-tweet dataset using the same context-aware sentiment rules used for individual tweets."
        )

    # Direct tweet analysis: extract quoted text when present; otherwise analyze the question itself.
    quoted = re.findall(r"[\"']([^\"']{8,})[\"']", q)
    direct_text = max(quoted, key=len) if quoted else q
    result = classify_sentiment(direct_text)

    # If the question asks for dataset examples, return matching classified tweets.
    if is_dataset_search_question(q) and not quoted:
        relevant = retrieve_relevant_tweets(q, df, top_k=8)
        analyzed_rows = []
        for _, row in relevant.iterrows():
            r = classify_sentiment(row["Tweet_Text"])
            analyzed_rows.append(
                f"- **{row['Tweet_ID']} — {r['sentiment']}**: {row['Tweet_Text']}"
            )
        if analyzed_rows:
            return "### Relevant Dataset Tweets\n\n" + "\n".join(analyzed_rows)

    # Simple follow-up handling.
    if history and q.lower() in {"why?", "why", "explain", "explain why"}:
        previous = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
        return previous + "\n\n**Explanation:** The classification is based on the overall wording, context, negation, and strength of sentiment signals rather than a single keyword."

    return (
        f"### Sentiment Analysis\n\n"
        f"**Sentiment:** {result['sentiment']}\n\n"
        f"**Reason:** {result['reason']}\n\n"
        f"**Confidence:** {result['confidence']}\n\n"
        f"**Sentiment Score:** `{result['score']}`\n\n"
        "*Analysis is performed locally by the UC03 NLP sentiment engine; no paid API is used.*"
    )


st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
st.title("💬 UC03 - Social Media Sentiment Analysis Agent")
st.caption(
    "Free, context-aware Positive / Negative / Neutral sentiment analysis grounded in the supplied synthetic tweet dataset."
)

try:
    dataset_path = locate_dataset()
    tweets = load_dataset(str(dataset_path))
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("⚙️ Agent Configuration")
    st.success("🟢 Free Local NLP Engine")
    st.write("API key: **Not required**")
    st.divider()
    st.metric("Tweets loaded", len(tweets))
    st.caption("Primary source: supplied synthetic Excel dataset")
    st.caption("No OpenAI API • No paid credits • No external model")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

chat_tab, dataset_tab, about_tab = st.tabs(["🤖 AI Agent", "📊 Dataset", "ℹ️ How It Works"])

with chat_tab:
    st.subheader("Ask the sentiment agent")
    examples = [
        "Select an example...",
        "What is the sentiment of: 'The new update is amazing! Everything feels much faster now.'?",
        "Analyze this tweet: 'The application keeps crashing every few minutes.'",
        "What is the sentiment of: 'I expected the update to be terrible, but honestly it works surprisingly well.'?",
        "Why can a tweet containing a negative word still be Neutral?",
        "Find the overall sentiment of the dataset.",
        "Which tweets appear to express negative sentiment about performance?",
    ]
    example = st.selectbox("Try a demonstration question", examples)
    if example != examples[0] and st.button("Use example"):
        st.session_state.prefill = example
        st.rerun()
    question_default = st.session_state.pop("prefill", "")
    with st.form("sentiment_form", clear_on_submit=False):
        question = st.text_area(
            "Your question", value=question_default, height=120,
            placeholder="Example: Analyze this tweet: 'I expected the update to be terrible, but it is surprisingly good.'",
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if submitted and question.strip():
        user_question = question.strip()
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sentiment..."):
                answer = answer_question(user_question, tweets, st.session_state.messages)
            st.markdown(answer)
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

with dataset_tab:
    st.subheader("Supplied Synthetic Tweet Dataset")
    st.write(f"Loaded **{len(tweets)} tweets** from `{DATASET_NAME}`.")
    analyzed, distribution = analyze_dataset(tweets)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tweets", len(tweets))
    c2.metric("Positive", f"{distribution['Positive']}%")
    c3.metric("Negative", f"{distribution['Negative']}%")
    c4.metric("Neutral", f"{distribution['Neutral']}%")
    st.dataframe(analyzed, use_container_width=True, hide_index=True)
    st.info("The supplied Excel dataset is the primary knowledge source. All sentiment analysis runs locally in Python.")

with about_tab:
    st.subheader("How UC03 works")
    st.markdown(
        """
**1. Dataset loading**  
The application loads the supplied Excel file containing the synthetic tweets.

**2. Lightweight retrieval**  
For dataset-focused questions, keyword overlap retrieves the most relevant tweets. This avoids an unnecessary vector database for a 100-row dataset.

**3. Local NLP sentiment engine**  
The agent uses an explainable polarity lexicon with phrase matching, negation handling, intensifiers, emojis, and contrast-word weighting.

**4. Context-aware classification**  
The engine considers the overall meaning before choosing Positive, Negative, or Neutral. For example, “not good” is treated differently from “good”, and content after “but” receives more weight.

**5. Overall analysis**  
When asked for overall dataset sentiment, every supplied tweet is classified and the application calculates counts and percentages directly.

**6. Free deployment**  
The app can run on Streamlit Community Cloud without an OpenAI API key, paid credits, or external model download.
        """
    )

st.divider()
st.caption("UC03 • Free local NLP sentiment agent • Dataset-grounded • No API key • No vector database • No generated database files")
