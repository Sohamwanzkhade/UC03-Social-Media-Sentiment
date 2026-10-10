"""
UC03 - Social Media Sentiment Analysis AI Agent
FINAL SUBMISSION FILE: code.py

Run locally:
    pip install -r requirements.txt
    streamlit run code.py

API key options (priority order):
    1. Streamlit secret: OPENAI_API_KEY
    2. Environment variable: OPENAI_API_KEY
    3. Session-only key entered in the sidebar

Dataset expected in the same folder:
    Synthetic Data - Social Media Tweets.xlsx

The app intentionally avoids unnecessary vector databases, local model files,
embeddings, caches, and generated database files. The supplied 100-tweet Excel
file is the primary knowledge source.
"""

from pathlib import Path
import os
import re
from typing import Dict, List

import pandas as pd
import streamlit as st
from openai import OpenAI


APP_TITLE = "UC03 - Social Media Sentiment Analysis Agent"
DATASET_NAME = "Synthetic Data - Social Media Tweets.xlsx"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

SYSTEM_PROMPT = """
You are UC03, a specialized Social Media Sentiment Analysis AI Agent.

Your only purpose is social-media sentiment analysis using the supplied tweet
information and the user's question.

Sentiment labels:
- Positive
- Negative
- Neutral

Rules:
1. Judge the overall meaning and context, not isolated keywords.
2. Consider negation, informal language, emojis, punctuation, contrast words,
   and contextual expressions.
3. A factual statement with no clear opinion is Neutral.
4. If positive and negative content are mixed, determine the overall expressed
   opinion rather than choosing a label from one keyword.
5. Do not invent facts, motives, events, or details not supported by the tweet
   or supplied dataset evidence.
6. The supplied synthetic tweet dataset is the primary knowledge source.
7. When a question refers to a dataset tweet, use matching dataset evidence when
   available and clearly distinguish dataset evidence from the user's own text.
8. For an individual tweet, return:
   Sentiment: Positive / Negative / Neutral
   Reason: concise explanation
   Confidence: High / Medium / Low
9. For multiple tweets, provide counts and percentages when the relevant tweets
   are identifiable.
10. For an overall dataset question, report total tweets, Positive count and %,
    Negative count and %, Neutral count and %, overall sentiment, and a short
    insight.
11. Follow-up questions are allowed, but stay within the UC03 sentiment scope.
12. Do not browse or rely on external information unless the application explicitly
    supplies external-source content. In this application, the dataset is the
    default source.
13. Keep answers concise, clear, professional, and suitable for a demonstration.
"""


def locate_dataset() -> Path:
    candidates = [
        Path(__file__).resolve().parent / DATASET_NAME,
        Path.cwd() / DATASET_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{DATASET_NAME}'. Put the supplied Excel file in the "
        "same folder as code.py."
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


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def retrieve_relevant_tweets(query: str, df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    query_terms = tokenize(query)
    if not query_terms:
        return df.head(top_k).copy()
    query_lower = query.lower().strip()
    scored = []
    for _, row in df.iterrows():
        tweet_text = row["Tweet_Text"]
        tweet_terms = tokenize(tweet_text)
        overlap = len(query_terms & tweet_terms)
        phrase_bonus = 3 if query_lower in tweet_text.lower() else 0
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


def get_api_key() -> str:
    session_key = st.session_state.get("runtime_api_key", "").strip()
    if session_key:
        return session_key
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        secret_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        secret_key = ""
    return secret_key


def get_client() -> OpenAI:
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to Streamlit Secrets "
            "or set the OPENAI_API_KEY environment variable."
        )
    return OpenAI(api_key=key)


def call_agent(client: OpenAI, model: str, user_question: str,
               evidence: str, conversation: List[Dict[str, str]]) -> str:
    history = ""
    if conversation:
        recent = conversation[-6:]
        history = "\n\nRecent conversation:\n" + "\n".join(
            f"{item['role'].upper()}: {item['content']}" for item in recent
        )
    prompt = f"""
User question:
{user_question}

Relevant dataset evidence:
{evidence}
{history}

Answer the user using the UC03 rules.
If the user supplied a tweet directly, analyze that tweet even if it is not in
this dataset. Do not claim that the user's tweet is a dataset record unless the
evidence proves it.
"""
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )
    return response.output_text.strip()


def analyze_overall_dataset(client: OpenAI, model: str, df: pd.DataFrame) -> str:
    dataset_text = format_tweets(df)
    prompt = f"""
Analyze ALL of the following supplied tweets for UC03.

Return exactly these sections:

Total Tweets: <number>
Positive: <count> (<percentage>%)
Negative: <count> (<percentage>%)
Neutral: <count> (<percentage>%)
Overall Sentiment: <Positive/Negative/Neutral>

Key Insight:
<2-4 concise sentences>

Requirements:
- Consider every supplied tweet.
- Use overall meaning and context.
- Consider negation, informal language, emojis, and mixed statements.
- Do not use outside information.
- Percentages must be based on the supplied dataset and add to approximately 100%.

Dataset:
{dataset_text}
"""
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )
    return response.output_text.strip()


def is_overall_question(question: str) -> bool:
    terms = [
        "overall sentiment", "overall analysis", "sentiment distribution",
        "all tweets", "entire dataset", "whole dataset", "dataset sentiment",
        "sentiment of the dataset", "analyze the dataset",
    ]
    q = question.lower()
    return any(term in q for term in terms)


st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
st.title("💬 UC03 - Social Media Sentiment Analysis Agent")
st.caption(
    "Context-aware Positive / Negative / Neutral sentiment analysis grounded in "
    "the supplied synthetic social-media tweet dataset."
)

try:
    dataset_path = locate_dataset()
    tweets = load_dataset(str(dataset_path))
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("⚙️ Agent Configuration")
    model = st.text_input(
        "OpenAI model", value=DEFAULT_MODEL,
        help="Use a model available to your OpenAI API account.",
    ).strip()
    runtime_key = st.text_input(
        "OpenAI API key (optional)", type="password",
        help="Session-only fallback. It is not saved by this application.",
    )
    if runtime_key.strip():
        st.session_state.runtime_api_key = runtime_key.strip()
    st.write("API key: **Configured**" if get_api_key() else "API key: **Not configured**")
    st.divider()
    st.metric("Tweets loaded", len(tweets))
    st.caption("Primary source: supplied synthetic Excel dataset")
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
            placeholder=("Example: Analyze this tweet: 'I expected the update to be terrible, "
                         "but it is surprisingly good.'"),
        )
        submitted = st.form_submit_button("Analyze", type="primary")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if submitted and question.strip():
        user_question = question.strip()
        with st.chat_message("user"):
            st.markdown(user_question)
        if not get_api_key():
            answer = (
                "The UC03 agent is ready, but no OpenAI API key is configured. "
                "Add `OPENAI_API_KEY` to Streamlit Secrets or enter a session-only key in the sidebar."
            )
        elif not model:
            answer = "Please enter an OpenAI model name in the sidebar."
        else:
            try:
                client = get_client()
                if is_overall_question(user_question):
                    with st.spinner("Analyzing the supplied dataset..."):
                        answer = analyze_overall_dataset(client, model, tweets)
                else:
                    evidence_df = retrieve_relevant_tweets(user_question, tweets, top_k=8)
                    evidence = format_tweets(evidence_df)
                    with st.spinner("Analyzing sentiment..."):
                        answer = call_agent(
                            client, model, user_question, evidence,
                            st.session_state.messages,
                        )
            except Exception as exc:
                answer = (
                    "I could not complete the analysis. Please check the OpenAI API "
                    f"key, selected model, and deployment configuration.\n\nError: `{exc}`"
                )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

with dataset_tab:
    st.subheader("Supplied Synthetic Tweet Dataset")
    st.write(f"Loaded **{len(tweets)} tweets** from `{DATASET_NAME}`.")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Tweets", len(tweets))
    with c2:
        st.metric("Columns", len(tweets.columns))
    st.dataframe(tweets, use_container_width=True, hide_index=True)
    st.info(
        "The supplied dataset is the primary knowledge source. The application "
        "does not automatically browse Google or other external websites."
    )

with about_tab:
    st.subheader("How UC03 works")
    st.markdown(
        """
**1. Dataset loading**  
The application loads the supplied Excel file containing the synthetic tweets.

**2. Lightweight retrieval**  
For a targeted question, a small keyword-overlap retriever selects relevant tweets. This avoids an unnecessary vector database for a 100-row dataset.

**3. OpenAI reasoning**  
The selected evidence and the user's question are sent to the OpenAI Responses API with the UC03 sentiment-analysis instructions.

**4. Context-aware classification**  
The agent considers context, negation, informal language, emojis, and mixed statements before choosing Positive, Negative, or Neutral.

**5. Overall analysis**  
When asked for the overall dataset sentiment, the agent receives all supplied tweets and returns the sentiment distribution and a concise insight.

**6. Deployment**  
The same `code.py` can be deployed on Streamlit Community Cloud. The OpenAI API key should be stored as a Streamlit secret named `OPENAI_API_KEY` rather than committed to GitHub.
        """
    )

st.divider()
st.caption(
    "UC03 • Dataset-grounded sentiment agent • No vector database • No local model files • No generated database files"
)
