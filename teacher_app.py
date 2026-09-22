import os
import re
import json
import collections
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Load Dynamic Course Spec ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "OCR A-Level Computer Science",
        "level": "A-Level"
    }

COURSE_TITLE = COURSE_SPEC.get("course_title", "OCR A-Level Computer Science")
LEVEL = COURSE_SPEC.get("level", "A-Level")

# --- Database Connection Helper ---
def get_db_engine():
    """Retrieves SQLAlchemy engine using Supabase credentials in secrets.toml."""
    try:
        db_url = st.secrets["postgres"]["url"]
        return create_engine(db_url, pool_pre_ping=True)
    except Exception as e:
        st.error(f"❌ Database secret configuration error: {e}")
        return None

# --- Page Setup & Dynamic Styling ---
st.set_page_config(page_title=f"{COURSE_TITLE} - Teacher Console", layout="wide", page_icon="📊")

THEME_PRIMARY = "#0284c7"
THEME_GRADIENT_START = "#0369a1"
THEME_GRADIENT_END = "#0284c7"

st.markdown(f"""
    <style>
    .stApp {{ background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
    div[data-testid="stSidebar"] {{ background-color: #ffffff; border-right: 1px solid #e2e8f0; }}
    h1, h2, h3 {{ color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }}
    .console-header {{ 
        background: linear-gradient(135deg, {THEME_GRADIENT_START} 0%, {THEME_GRADIENT_END} 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.4em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.25);
        margin-bottom: 24px;
    }}
    .metric-card {{
        background: #ffffff; padding: 18px; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; text-align: center;
    }}
    .afl-callout {{
        background: #f0f9ff; border-left: 5px solid {THEME_PRIMARY}; padding: 16px 20px;
        border-radius: 8px; color: #0369a1; font-size: 0.98em; line-height: 1.5; margin-top: 15px;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="console-header">📊 {COURSE_TITLE} ({LEVEL}) Teacher Analytics Console</div>', unsafe_allow_html=True)

# --- Database Fetch Helper ---
@st.cache_data(ttl=60)
def load_data():
    engine = get_db_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        query = "SELECT * FROM activity_logs ORDER BY timestamp DESC;"
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data from Supabase: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.warning("⚠️ No analytics data found in Supabase. Please run `generate_test_data.py` first to populate test logs.")
    st.stop()

# --- Dynamic Filter Generation ---
st.sidebar.header("🎯 Filter Cohort Data")

if st.sidebar.button("🔄 Refresh Analytics Data"):
    st.cache_data.clear()
    st.rerun()

subjects = ["All"] + sorted(list(df_raw["subject"].unique()))
selected_subject = st.sidebar.selectbox("Filter by Subject / Branch:", options=subjects)

filtered_df = df_raw.copy()
if selected_subject != "All":
    filtered_df = filtered_df[filtered_df["subject"] == selected_subject]

units = ["All"] + sorted(list(filtered_df["unit"].unique()))
selected_unit = st.sidebar.selectbox("Filter by Unit:", options=units)

if selected_unit != "All":
    filtered_df = filtered_df[filtered_df["unit"] == selected_unit]

modes = ["All"] + sorted(list(filtered_df["app_mode"].unique()))
selected_mode = st.sidebar.selectbox("Filter by Activity Mode:", options=modes)

if selected_mode != "All":
    filtered_df = filtered_df[filtered_df["app_mode"] == selected_mode]

st.sidebar.write("---")
st.sidebar.caption("🔒 **Zero-PII Compliance:** Data is logged statelessly at session level. No individual pupil identities or tokens are stored.")

# --- Macro KPI Cards ---
total_sessions = len(filtered_df)
avg_score = round(filtered_df["score_pct"].mean(), 1) if total_sessions > 0 else 0.0
misconception_count = int(filtered_df["misconception_flag"].sum()) if total_sessions > 0 else 0

all_missed_keywords = []
for kw_raw in filtered_df["keywords_missed"].dropna():
    if not kw_raw:
        continue
    # Support both comma-separated strings and JSON arrays
    if kw_raw.startswith("["):
        try:
            kw_list = json.loads(kw_raw)
            all_missed_keywords.extend([k.strip() for k in kw_list if k.strip()])
        except Exception:
            pass
    else:
        all_missed_keywords.extend([k.strip() for k in kw_raw.split(",") if k.strip()])

kw_counter = collections.Counter(all_missed_keywords)
top_missed_term = kw_counter.most_common(1)[0][0] if kw_counter else "None"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><h3>{total_sessions}</h3><p>Total Practice Sessions</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><h3>{avg_score}%</h3><p>Cohort Avg Score</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><h3>{misconception_count}</h3><p>Misconception Flags</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><h3>{top_missed_term}</h3><p>Top Missed Term</p></div>', unsafe_allow_html=True)

st.write("")

# --- Visualizations ---
tab1, tab2 = st.tabs(["📌 Keyword Gap Analysis", "📈 Engagement & Performance"])

with tab1:
    st.subheader("Top Specification Keywords Missed Across Cohort")
    if kw_counter:
        top_kws = pd.DataFrame(kw_counter.most_common(10), columns=["Keyword", "Frequency"])
        fig_kw = px.bar(
            top_kws, x="Frequency", y="Keyword", orientation="h",
            color="Frequency", color_continuous_scale="Reds",
            title="A-Level Specification Terms Requiring Targeted Remediation"
        )
        fig_kw.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_kw, use_container_width=True)
        
        st.markdown(f"""
            <div class="afl-callout">
                💡 <b>Actionable Assessment for Learning (AfL) Insight:</b><br>
                The most frequently missed technical term in this view is <b>"{top_missed_term}"</b> (missed <b>{kw_counter[top_missed_term]}</b> times across student practice rounds). 
                Consider opening your next lesson with a 5-minute retrieval starter or whiteboard trace task targeting this specific Computer Science concept.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No missed keywords recorded for this selection.")

with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Activity Mode Distribution")
        mode_counts = filtered_df["app_mode"].value_counts().reset_index()
        mode_counts.columns = ["Mode", "Sessions"]
        fig_mode = px.pie(mode_counts, names="Mode", values="Sessions", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_mode, use_container_width=True)
        
    with col_right:
        st.subheader("Performance Score Distribution")
        fig_score = px.histogram(filtered_df, x="score_pct", nbins=10, title="Cohort Score Range (%)", color_discrete_sequence=[THEME_PRIMARY])
        st.plotly_chart(fig_score, use_container_width=True)

st.write("---")

# --- Interactive AI Chatbot (Multi-Turn Conversational Analytics) ---
st.subheader("💬 Interactive AI Telemetry Assistant")

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

col_info, col_clear = st.columns([5, 1])
with col_clear:
    if st.button("🗑️ Reset Chat"):
        st.session_state.ai_chat_history = []
        st.rerun()

def clean_html_formatting(text):
    """Strips raw HTML tags like <u> to ensure clean copy-pasting."""
    return re.sub(r'</?[a-zA-Z0-9]+[^>]*>', '', text)

def ask_ai_about_data(user_question, chat_history):
    schema_info = """
    Table name: activity_logs (PostgreSQL)
    Columns:
    - id (SERIAL PRIMARY KEY)
    - timestamp (TIMESTAMPTZ)
    - subject (TEXT) - e.g., 'OCR A-Level Computer Science'
    - level (TEXT) - e.g., 'A-Level'
    - unit (TEXT) - e.g., '1.1 The characteristics of contemporary processors...'
    - subtopic (TEXT) - e.g., '1.1.1 Structure and function of the processor'
    - app_mode (TEXT) - 'socratic', 'quiz', 'extended', 'rewrite'
    - score_pct (REAL) - percentage score (0.0 to 100.0)
    - keywords_used (TEXT) - comma-separated list of technical CS terms used correctly
    - keywords_missed (TEXT) - comma-separated list of technical CS terms missed
    - misconception_flag (INTEGER) - 1 if fundamental CS misconception detected (<45%), else 0
    """

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

    full_transcript = ""
    for msg in chat_history:
        full_transcript += f"Teacher: {msg['question']}\nAI Assistant: {msg['summary']}\n\n"

    # --- Step 1: Classify Intent ---
    intent_prompt = f"""
    Analyze the teacher's latest message based on the conversation transcript.
    
    Transcript:
    {full_transcript}
    
    Latest Message: "{user_question}"
    
    Is the teacher asking to:
    A) Query/fetch NEW data from the database (e.g., scores, counts, top missed CS terms, subtopic search)?
    B) Refine, reformat, expand, or follow up on previously generated text/questions (e.g., remove options, format answers, bold terms, rephrase response)?
    
    Respond with ONLY the single letter 'A' or 'B'.
    """
    
    intent_resp = llm.invoke(intent_prompt)
    intent_raw = intent_resp.content
    intent_text = "".join([str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in intent_raw]) if isinstance(intent_raw, list) else str(intent_raw)
    intent = intent_text.strip().upper()

    # --- PATH B: Refinement / Content Follow-up (No SQL Executed) ---
    if "B" in intent and chat_history:
        refine_prompt = f"""
        You are an expert A-Level Computer Science pedagogical assistant.
        
        Full Conversation History:
        {full_transcript}
        
        Teacher's Refinement Request: {user_question}
        
        CRITICAL FORMATTING INSTRUCTION:
        Do NOT output raw HTML tags (e.g., do NOT use <u>, <i>, <b>). Use standard Markdown bolding (**key term**) for emphasis.

        Directly address the request by building on the EXACT questions or text generated in the previous assistant message.
        Do NOT query new database terms. Maintain complete continuity with the previous output.
        """
        
        refine_resp = llm.invoke(refine_prompt)
        refine_content = refine_resp.content
        summary = "".join([str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in refine_content]) if isinstance(refine_content, list) else str(refine_content)
        summary = clean_html_formatting(summary)
        
        last_sql = chat_history[-1].get("sql", "N/A (Refinement of previous response)")
        last_df = chat_history[-1].get("df", pd.DataFrame())
        return summary, last_df, f"-- [Refinement Task: Reusing previous context]\n-- Previous SQL:\n{last_sql}"

    # --- PATH A: New Data Retrieval (Text-to-SQL for PostgreSQL) ---
    sql_prompt = f"""
    You are an expert PostgreSQL data analyst for an A-Level Computer Science department.
    Given the PostgreSQL table schema below and previous context, write a SINGLE valid, read-only SELECT query to answer the teacher's latest question.
    Use PostgreSQL syntax. Do NOT include markdown fences (like ```sql), code blocks, or commentary—output ONLY the plain SQL query text.

    Schema:
    {schema_info}

    Latest Question: {user_question}
    """
    
    response_obj = llm.invoke(sql_prompt)
    raw_content = response_obj.content
    raw_text = "".join([str(block.get("text", block)) if isinstance(block, dict) else str(block) for block in raw_content]) if isinstance(raw_content, list) else str(raw_text)

    sql_query = raw_text.strip()
    if sql_query.startswith("```"):
        sql_query = sql_query.split("\n", 1)[-1]
    if sql_query.endswith("```"):
        sql_query = sql_query.rsplit("```", 1)[0]
    sql_query = sql_query.replace("```sql", "").strip()

    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            df_result = pd.read_sql_query(text(sql_query), conn)
    except Exception as e:
        return f"❌ Query execution error: `{sql_query}`\n\nDetails: {str(e)}", None, sql_query

    synthesis_prompt = f"""
    You are a Head of Computer Science evaluating learning analytics and generating targeted A-Level revision resources.
    Synthesize the query results into a concise, professional response that directly fulfills the teacher's request.
    
    CRITICAL FORMATTING INSTRUCTION:
    Do NOT use HTML tags (e.g., <u>, <b>). Use standard Markdown bolding (**term**) for key terms.

    Teacher Question: {user_question}
    Query Results:
    {df_result.to_string(index=False)}
    """
    
    syn_response = llm.invoke(synthesis_prompt)
    syn_content = syn_response.content
    summary = "".join([str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in syn_content]) if isinstance(syn_content, list) else str(syn_content)
    summary = clean_html_formatting(summary)

    return summary, df_result, sql_query

# --- Render Active Chat Stream ---
for idx, entry in enumerate(st.session_state.ai_chat_history):
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(f"**AI Insight:**\n\n{entry['summary']}")
        
        st.download_button(
            label="📥 Download Generated Starter / Resource (.txt)",
            data=entry["summary"],
            file_name=f"retrieval_starter_{idx + 1}.txt",
            mime="text/plain",
            key=f"dl_{idx}"
        )
        
        with st.expander("📊 View SQL Query & Raw Data Table"):
            st.code(entry["sql"], language="sql")
            st.dataframe(entry["df"], use_container_width=True)

# --- Chat Input for Questions & Follow-ups ---
teacher_query = st.chat_input("Ask an initial or follow-up question about A-Level cohort telemetry...")

if teacher_query:
    with st.chat_message("user"):
        st.write(teacher_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Supabase database..."):
            summary, df_res, sql_used = ask_ai_about_data(teacher_query, st.session_state.ai_chat_history)
            
            if df_res is not None:
                st.markdown(f"**AI Insight:**\n\n{summary}")
                
                st.download_button(
                    label="📥 Download Generated Starter / Resource (.txt)",
                    data=summary,
                    file_name=f"retrieval_starter_{len(st.session_state.ai_chat_history) + 1}.txt",
                    mime="text/plain",
                    key=f"dl_live_{len(st.session_state.ai_chat_history)}"
                )
                
                with st.expander("📊 View SQL Query & Raw Data Table"):
                    st.code(sql_used, language="sql")
                    st.dataframe(df_res, use_container_width=True)
                
                st.session_state.ai_chat_history.append({
                    "question": teacher_query,
                    "summary": summary,
                    "sql": sql_used,
                    "df": df_res
                })
            else:
                st.error(summary)