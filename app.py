import importlib
import json
import os
import re
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Ensure API Key is set BEFORE loading socratic_fsm
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

import socratic_fsm
importlib.reload(socratic_fsm)

from socratic_fsm import (
    workflow,
    generate_quiz_questions,
    evaluate_quiz_answers,
    generate_extended_question,
    grade_extended_response,
    generate_layman_transformation_prompt,
    grade_disciplinary_rewrite
)

# --- Load Dynamic Course Spec ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "OCR A-Level Computer Science",
        "level": "A-Level",
        "target_turns": 7,
        "topics": {"General": ["General Practice"]}
    }

COURSE_TITLE = COURSE_SPEC.get("course_title", "OCR A-Level Computer Science")
LEVEL = COURSE_SPEC.get("level", "A-Level")
TARGET_TURNS = COURSE_SPEC.get("target_turns", 7)

# --- COLOR THEME CONFIGURATION: OPTION SUBJECT (Forest & Emerald Green) ---
THEME_PRIMARY = "#059669"        # Emerald Green for primary buttons, borders, and progress bars
THEME_GRADIENT_START = "#065f46"  # Dark Forest Green Header Start
THEME_GRADIENT_END = "#059669"    # Emerald Green Header End
THEME_BANNER_BG = "#ecfdf5"       # Light Emerald Purpose Banner
THEME_BANNER_BORDER = "#a7f3d0"   # Soft Emerald Border
THEME_BANNER_TEXT = "#065f46"     # Dark Emerald Banner Text

# --- Database & Zero-PII Logging Setup ---
Base = declarative_base()

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    subject = Column(Text, nullable=False)
    level = Column(Text, nullable=False)
    unit = Column(Text, nullable=False)
    subtopic = Column(Text, nullable=False)
    app_mode = Column(Text, nullable=False)
    score_pct = Column(Float, nullable=False)
    keywords_used = Column(Text, nullable=True)
    keywords_missed = Column(Text, nullable=True)
    misconception_flag = Column(Integer, default=0)

def log_session_to_supabase(unit: str, subtopic: str, app_mode: str, score_pct: float, keywords_used: list = None, keywords_missed: list = None, misconception_flag: int = 0):
    """Logs session metrics statelessly to Supabase activity_logs table."""
    try:
        db_url = st.secrets["postgres"]["url"]
        engine = create_engine(db_url, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        kw_used_str = ", ".join([str(k) for k in keywords_used if k]) if keywords_used else None
        kw_missed_str = ", ".join([str(k) for k in keywords_missed if k]) if keywords_missed else None

        log_entry = ActivityLog(
            subject=COURSE_TITLE,
            level=LEVEL,
            unit=unit or "General",
            subtopic=subtopic or "General Practice",
            app_mode=app_mode,
            score_pct=round(float(score_pct), 1),
            keywords_used=kw_used_str,
            keywords_missed=kw_missed_str,
            misconception_flag=int(misconception_flag)
        )

        session.add(log_entry)
        session.commit()
        session.close()
    except Exception as e:
        st.warning(f"⚠️ Telemetry log failed: {e}")

# --- Formatting Helpers ---
def extract_clean_text(response) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return extract_clean_text(response.content)
    if isinstance(response, list) and len(response) > 0:
        first_item = response[0]
        if isinstance(first_item, dict):
            return first_item.get("text", str(first_item))
        elif hasattr(first_item, "text"):
            return first_item.text
        return extract_clean_text(first_item)
    if isinstance(response, dict):
        if "text" in response:
            return response["text"]
        elif "content" in response:
            return extract_clean_text(response["content"])
    return str(response)

def clean_latex(text: str) -> str:
    text = re.sub(r'\$([^\$]+)\$', r'<b>\1</b>', text)
    return text.replace('$', '')

def md_to_html(text: str) -> str:
    text = re.sub(r'^####\s+(.*$)', r'<h4 style="margin: 4px 0 1px 0; font-size: 1.05em; color: inherit;">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*$)', r'<h3 style="margin: 6px 0 2px 0; font-size: 1.1em; color: inherit;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*$)', r'<h2 style="margin: 8px 0 2px 0; font-size: 1.2em; color: inherit;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^\s*[-*]\s+(.*$)', r'<div style="margin: 1px 0;">• \1</div>', text, flags=re.MULTILINE)
    text = re.sub(r'(</(div|h2|h3|h4)>)\s*\n+', r'\1', text)
    text = re.sub(r'\n{2,}', '<br>', text)
    text = text.replace('\n', '<br>')
    return re.sub(r'(<br\s*/?>\s*)+', '<br>', text)

# --- CSS Styling ---
st.markdown(f"""
    <style>
    .stApp {{ background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
    div[data-testid="stSidebar"] {{ background-color: #ffffff; border-right: 1px solid #e2e8f0; }}
    div[data-testid="stProgress"] > div > div > div {{ background-color: {THEME_PRIMARY} !important; }}
    h1, h2, h3 {{ color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }}
    .chat-header {{ 
        background: linear-gradient(135deg, {THEME_GRADIENT_START} 0%, {THEME_GRADIENT_END} 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.3em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(5, 150, 105, 0.25);
        margin-bottom: 20px;
    }}
    .purpose-banner {{
        background: {THEME_BANNER_BG}; border: 1px solid {THEME_BANNER_BORDER}; padding: 14px 18px;
        border-radius: 12px; color: {THEME_BANNER_TEXT}; font-size: 0.95em; margin-bottom: 20px;
        line-height: 1.4;
    }}
    .tutor-msg {{ 
        background-color: #ffffff; color: #1e293b; padding: 14px 18px; 
        border-radius: 18px 18px 18px 4px; margin-bottom: 12px; max-width: 82%; 
        line-height: 1.35; border: 1px solid #e2e8f0;
    }}
    .student-msg {{ 
        background: linear-gradient(135deg, {THEME_GRADIENT_START} 0%, {THEME_GRADIENT_END} 100%); 
        color: white; padding: 14px 18px; border-radius: 18px 18px 4px 18px; 
        margin-bottom: 12px; max-width: 82%; margin-left: auto; line-height: 1.35;
    }}
    .summary-box {{ 
        background: #fefce8; border-left: 5px solid #eab308; padding: 10px 14px; 
        border-radius: 12px; color: #713f12; font-size: 0.93em; margin: 8px 0; max-width: 85%; 
    }}
    .stButton > button {{ border-radius: 12px !important; font-weight: 600 !important; margin-top: 4px !important; margin-bottom: 4px !important; }}
    button[kind="primary"] {{
        background-color: {THEME_PRIMARY} !important;
        border-color: {THEME_PRIMARY} !important;
    }}
    div[data-baseweb="textarea"], 
    div[data-baseweb="textarea"] > div,
    textarea {{
        background-color: #ffffff !important;
    }}
    div[data-baseweb="textarea"] {{
        border: 2px solid {THEME_PRIMARY} !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }}
    div[data-baseweb="textarea"]:focus-within {{
        border-color: {THEME_GRADIENT_START} !important;
        box-shadow: 0 0 0 3px rgba(6, 95, 70, 0.3) !important;
    }}
    textarea {{
        color: #0f172a !important;
        font-size: 1rem !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialisation ---
if "active_unit" not in st.session_state:
    st.session_state.active_unit = None
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None
if "extended_question" not in st.session_state:
    st.session_state.extended_question = None
if "extended_results" not in st.session_state:
    st.session_state.extended_results = None
if "submitted_extended_answer" not in st.session_state:
    st.session_state.submitted_extended_answer = ""
if "rewrite_data" not in st.session_state:
    st.session_state.rewrite_data = None
if "rewrite_results" not in st.session_state:
    st.session_state.rewrite_results = None
if "student_rewrite_submission" not in st.session_state:
    st.session_state.student_rewrite_submission = ""
if "logged_sessions" not in st.session_state:
    st.session_state.logged_sessions = set()

if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [], 
        "sub_topic": st.session_state.active_topic, 
        "turn_count": 0, 
        "is_final_turn": False
    }

def reset_session():
    st.session_state.active_unit = None
    st.session_state.active_topic = None
    st.session_state.app_mode = None
    st.session_state.messages = []
    st.session_state.quiz_questions = None
    st.session_state.quiz_feedback = None
    st.session_state.extended_question = None
    st.session_state.extended_results = None
    st.session_state.submitted_extended_answer = ""
    st.session_state.rewrite_data = None
    st.session_state.rewrite_results = None
    st.session_state.student_rewrite_submission = ""
    st.session_state.logged_sessions = set()
    st.session_state.graph_state = {
        "messages": [], "sub_topic": None, "turn_count": 0, "is_final_turn": False
    }
    st.rerun()

# --- Dynamic Screen Router ---
if st.session_state.active_topic is None:
    st.markdown(f'<div class="chat-header">💻 {COURSE_TITLE} Socratic Coach</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="purpose-banner">
            💡 <b>App Purpose:</b> The aim of this app is to help you develop your use of disciplinary literacy. 
            To gain a top grade in Computer Science, you need to think, trace, and write like a Computer Scientist!
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("⚙️ Select Revision Target")
    
    raw_structure = COURSE_SPEC.get("subjects") or COURSE_SPEC.get("topics", {})
    first_key = next(iter(raw_structure), None)
    is_three_tier = isinstance(raw_structure.get(first_key), dict) if first_key else False

    if is_three_tier:
        st.write("Choose a paper, section, and subtopic to begin your practice session:")
        sel_subject = st.selectbox("🖥️ Step 1: Choose Component / Paper:", options=list(raw_structure.keys()))
        units_dict = raw_structure.get(sel_subject, {})
        sel_unit = st.selectbox("📘 Step 2: Choose Section / Unit:", options=list(units_dict.keys()))
        subtopics = units_dict.get(sel_unit, [])
        sel_subtopic = st.selectbox("🔍 Step 3: Choose Specific Subtopic:", options=subtopics)
        
        target_unit_name = sel_unit
        target_subtopic_name = sel_subtopic
        full_query = f"{sel_subject} - {sel_unit}: {sel_subtopic}"
    else:
        st.write("Choose a unit and subtopic to begin your practice session:")
        sel_unit = st.selectbox("📘 Step 1: Choose Unit / Component:", options=list(raw_structure.keys()))
        subtopics = raw_structure.get(sel_unit, [])
        sel_subtopic = st.selectbox("🔍 Step 2: Choose Specific Subtopic:", options=subtopics)
        
        target_unit_name = sel_unit
        target_subtopic_name = sel_subtopic
        full_query = sel_subtopic

    st.write("")

    if st.button("🤖 Start a Socratic Session", type="primary", use_container_width=True):
        st.session_state.app_mode = "socratic"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        st.session_state.graph_state["sub_topic"] = full_query
        st.rerun()

    if st.button("📑 Take a Retrieval Quiz", use_container_width=True):
        st.session_state.app_mode = "quiz"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        with st.spinner("Generating specification retrieval questions..."):
            st.session_state.quiz_questions = generate_quiz_questions(
                full_query, COURSE_TITLE, LEVEL
            )
        st.rerun()

    if st.button("⚡ Answer an Extended Question", use_container_width=True):
        st.session_state.app_mode = "extended"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        with st.spinner("Generating high-tier extended response scenario..."):
            st.session_state.extended_question = generate_extended_question(full_query, COURSE_TITLE, LEVEL)
        st.rerun()

    if st.button("🛠️ Rewrite an Answer", use_container_width=True):
        st.session_state.app_mode = "rewrite"
        st.session_state.active_unit = target_unit_name
        st.session_state.active_topic = target_subtopic_name
        with st.spinner("Generating informal response scenario..."):
            st.session_state.rewrite_data = generate_layman_transformation_prompt(full_query, COURSE_TITLE, LEVEL)
        st.rerun()

# --- Socratic Mode View ---
elif st.session_state.app_mode == "socratic":
    st.markdown(f'<div class="chat-header">🤖 {COURSE_TITLE} Coach</div>', unsafe_allow_html=True)
    
    student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")

    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.metric(label="Turn Counter", value=f"{student_turns} / {TARGET_TURNS}")
        st.progress(min(student_turns / TARGET_TURNS, 1.0))
        st.write("---")
        if st.button("🔄 New Session / Change Topic", use_container_width=True):
            reset_session()

    if len(st.session_state.messages) == 0:
        initial_greeting = (
            f"Welcome! We're exploring **{st.session_state.active_topic}** today. "
            f"To get started, what key data structure, algorithm, hardware principle, or theory concept would you like to review?"
        )
        st.session_state.messages.append({"role": "tutor", "content": initial_greeting, "style": "tutor-msg"})
        st.session_state.graph_state["messages"].append(AIMessage(content=initial_greeting))

    for msg in st.session_state.messages:
        html_content = md_to_html(msg["content"])
        if msg["role"] == "tutor":
            div_class = msg.get("style", "tutor-msg")
            header = "💡 <b>Summary Note</b>" if div_class == "summary-box" else "🎓 <b>Tutor</b>"
            st.markdown(f'<div class="{div_class}">{header}<br><br>{html_content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="student-msg">🎒 <b>Student</b><br><br>{html_content}</div>', unsafe_allow_html=True)

    if student_turns >= TARGET_TURNS:
        st.info(f"🎉 **Session Complete!** You completed all {TARGET_TURNS} turns of the {LEVEL} Socratic dialogue.")

    is_disabled = student_turns >= TARGET_TURNS
    placeholder = "Session complete. Select a new topic in the sidebar." if is_disabled else "Type your response here..."
    
    if user_input := st.chat_input(placeholder, disabled=is_disabled):
        st.session_state.messages.append({"role": "student", "content": user_input})
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        current_student_turns = sum(1 for m in st.session_state.messages if m.get("role") == "student")
        st.session_state.graph_state["turn_count"] = current_student_turns
        st.session_state.graph_state["is_final_turn"] = (current_student_turns >= TARGET_TURNS)

        with st.spinner("Analyzing response and generating feedback..."):
            input_payload = {
                "messages": st.session_state.graph_state["messages"],
                "sub_topic": st.session_state.active_topic,
                "turn_count": current_student_turns,
                "is_final_turn": (current_student_turns >= TARGET_TURNS)
            }
            updated_state = workflow.invoke(input_payload)
        
        last_msg = updated_state["messages"][-1]
        ai_reply = clean_latex(extract_clean_text(last_msg))

        split_match = re.split(r'={3,}\s*SPLIT\s*={3,}', ai_reply, flags=re.IGNORECASE)
        if len(split_match) > 1:
            st.session_state.messages.append({"role": "tutor", "content": split_match[0].strip(), "style": "tutor-msg"})
            st.session_state.messages.append({"role": "tutor", "content": split_match[1].strip(), "style": "summary-box"})
            
            # --- DATABASE LOGGING FOR SOCRATIC COMPLETE ---
            if "socratic_completed" not in st.session_state.logged_sessions:
                score_match = re.search(r'Overall Accuracy:\s*(\d+)%', split_match[0])
                score_pct = float(score_match.group(1)) if score_match else 70.0
                
                kw_used_match = re.search(r'Keywords Used Well:\s*\[(.*?)\]', split_match[0])
                kw_missed_match = re.search(r'Missed Terms to Learn:\s*\[(.*?)\]', split_match[0])
                
                used_list = [k.strip() for k in kw_used_match.group(1).split(',')] if kw_used_match else []
                missed_list = [k.strip() for k in kw_missed_match.group(1).split(',')] if kw_missed_match else []
                
                log_session_to_supabase(
                    unit=st.session_state.active_unit,
                    subtopic=st.session_state.active_topic,
                    app_mode="socratic",
                    score_pct=score_pct,
                    keywords_used=used_list,
                    keywords_missed=missed_list,
                    misconception_flag=1 if score_pct < 45.0 else 0
                )
                st.session_state.logged_sessions.add("socratic_completed")
        else:
            st.session_state.messages.append({"role": "tutor", "content": ai_reply, "style": "tutor-msg"})

        st.session_state.graph_state = updated_state
        st.rerun()

# --- Quiz Mode View ---
elif st.session_state.app_mode == "quiz":
    st.markdown(f'<div class="chat-header">📑 {COURSE_TITLE} Retrieval Quiz</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.write("---")
        if st.button("🔄 Change Topic / Mode", use_container_width=True):
            reset_session()

    questions = st.session_state.get("quiz_questions")
    
    if questions:
        if st.session_state.quiz_feedback is None:
            with st.form("retrieval_quiz_form"):
                st.subheader(f"Practice Quiz: {st.session_state.active_topic}")
                user_answers = {}
                for idx, q in enumerate(questions, 1):
                    q_text = q.get("question", q) if isinstance(q, dict) else q
                    st.markdown(f"**Q{idx}: {q_text}**")
                    user_answers[idx] = st.text_input(f"Your Answer for Q{idx}:", key=f"quiz_ans_{idx}")
                    st.write("")
                
                submitted = st.form_submit_button("Submit Quiz for Feedback", type="primary", use_container_width=True)
                
                if submitted:
                    with st.spinner("Evaluating your responses against specification mark schemes..."):
                        feedback = evaluate_quiz_answers(
                            questions=questions,
                            user_answers=user_answers,
                            topic=st.session_state.active_topic,
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.quiz_feedback = feedback
                        
                        # --- DATABASE LOGGING FOR QUIZ ---
                        total_score = feedback.get("total_score", 0)
                        score_pct = (total_score / len(questions)) * 100.0 if questions else 0.0
                        
                        all_used, all_missed = [], []
                        for item in feedback.get("breakdown", []):
                            all_used.extend(item.get("keywords_used", []))
                            all_missed.extend(item.get("keywords_missed", []))
                            
                        log_session_to_supabase(
                            unit=st.session_state.active_unit,
                            subtopic=st.session_state.active_topic,
                            app_mode="quiz",
                            score_pct=score_pct,
                            keywords_used=all_used,
                            keywords_missed=all_missed,
                            misconception_flag=1 if score_pct < 45.0 else 0
                        )
                        st.rerun()
        else:
            feedback_data = st.session_state.quiz_feedback
            total_score = feedback_data.get("total_score", 0)
            breakdown = feedback_data.get("breakdown", [])
            total_questions = len(breakdown) if breakdown else len(questions)

            st.success(f"🎉 **Quiz Complete! Total Score: {total_score} / {total_questions}**\n\nReview your keyword accuracy breakdown below:")
            st.write("")

            for item in breakdown:
                q_num = item.get("question_num", "")
                q_text = item.get("question", "")
                score = item.get("score", 0)
                user_ans = item.get("student_answer", "No answer provided")
                model_ans = item.get("model_answer", "")
                used = ", ".join(item.get("keywords_used", [])) or "None"
                missed = ", ".join(item.get("keywords_missed", [])) or "None"
                explanation = item.get("explanation", "")

                label = f"Q{q_num}: {q_text} — Score: {score}/1"
                
                with st.expander(label, expanded=False):
                    st.markdown(f"**Your Answer:**\n\n> {user_ans}")
                    st.markdown(f"**Model Answer:** {model_ans}")
                    st.markdown(f"**Key Terms Used:** {used}")
                    st.markdown(f"**Missed Keywords:** {missed}")
                    st.info(f"💡 **Examiner Note:** {explanation}")

            st.write("")
            if st.button("🔄 Retake Quiz / Try Another Topic", type="primary", use_container_width=True):
                reset_session()
    else:
        st.error("No questions were generated. Please return and select a topic again.")
        if st.button("Back to Selection Screen"):
            reset_session()

# --- Extended Question Mode View ---
elif st.session_state.app_mode == "extended":
    st.markdown(f'<div class="chat-header">⚡ {COURSE_TITLE} Extended Question</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.write("---")
        if st.button("🔄 Change Topic / Mode", use_container_width=True):
            reset_session()

    q_text = st.session_state.get("extended_question")
    
    if q_text:
        if st.session_state.extended_results is None:
            st.subheader("Extended Response Challenge")
            st.markdown(f"**Question:** {q_text}")
            
            user_response = st.text_area("Write your detailed evaluation below (include trade-off analysis and technical terms):", height=200)
            
            if st.button("Submit Extended Answer", type="primary", use_container_width=True):
                if user_response.strip():
                    st.session_state.submitted_extended_answer = user_response.strip()
                    with st.spinner("Evaluating disciplinary literacy and complexity analysis..."):
                        results = grade_extended_response(
                            sub_topic=st.session_state.active_topic,
                            question=q_text,
                            student_answer=user_response.strip(),
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.extended_results = results
                        
                        # --- DATABASE LOGGING FOR EXTENDED QUESTION ---
                        score = results.get("score", 0)
                        max_score = results.get("max_score", 9)
                        score_pct = (score / max_score) * 100.0 if max_score > 0 else 0.0
                        
                        log_session_to_supabase(
                            unit=st.session_state.active_unit,
                            subtopic=st.session_state.active_topic,
                            app_mode="extended",
                            score_pct=score_pct,
                            keywords_used=results.get("keywords_used", []),
                            keywords_missed=results.get("keywords_missed", []),
                            misconception_flag=1 if score_pct < 45.0 else 0
                        )
                        st.rerun()
                else:
                    st.warning("Please type an answer before submitting.")
        else:
            res = st.session_state.extended_results
            
            with st.expander("📝 Your Submitted Extended Answer", expanded=True):
                st.markdown(f"**Question:** {q_text}")
                st.markdown(f"**Your Answer:**\n\n> {st.session_state.get('submitted_extended_answer', '')}")

            st.success(f"🎉 **Evaluation Complete! Score: {res.get('score', 0)} / {res.get('max_score', 9)} ({res.get('disciplinary_level', 'Developing')})**")
            
            st.markdown(f"**Strengths:** {res.get('strengths', '')}")
            st.markdown(f"**Key Terms Used:** {', '.join(res.get('keywords_used', [])) or 'None'}")
            st.markdown(f"**Missed Key Terms:** {', '.join(res.get('keywords_missed', [])) or 'None'}")
            st.info(f"💡 **Advice for Improvement:** {res.get('struggle_advice', '')}")
            
            with st.expander("📖 View Exemplar Top-Band Model Answer"):
                st.markdown(res.get("model_answer", ""))
                
            st.write("")
            if st.button("🔄 Try Another Question / Topic", type="primary", use_container_width=True):
                reset_session()

# --- Rewrite Mode View ---
elif st.session_state.app_mode == "rewrite":
    st.markdown(f'<div class="chat-header">🛠️ {COURSE_TITLE} Disciplinary Rewrite</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📌 Active Target")
        st.info(f"**Unit:** {st.session_state.active_unit}\n\n**Topic:** {st.session_state.active_topic}")
        st.write("---")
        if st.button("🔄 Change Topic / Mode", use_container_width=True):
            reset_session()

    r_data = st.session_state.get("rewrite_data")
    
    if r_data:
        if st.session_state.rewrite_results is None:
            st.subheader("Upgrade Informal Answer Challenge")
            st.markdown(f"**Question:** {r_data.get('question')}")
            st.warning(f"**Informal Student Draft:**\n\n\"{r_data.get('layman_answer')}\"")
            
            student_rewrite = st.text_area("Rewrite this response using precise Computer Science terminology and complexity notation:", height=180)
            
            if st.button("Submit Upgraded Rewrite", type="primary", use_container_width=True):
                if student_rewrite.strip():
                    st.session_state.student_rewrite_submission = student_rewrite.strip()
                    with st.spinner("Grading terminology upgrade..."):
                        results = grade_disciplinary_rewrite(
                            sub_topic=st.session_state.active_topic,
                            question=r_data.get('question'),
                            layman_answer=r_data.get('layman_answer'),
                            student_rewrite=student_rewrite.strip(),
                            course_title=COURSE_TITLE,
                            level=LEVEL
                        )
                        st.session_state.rewrite_results = results
                        
                        # --- DATABASE LOGGING FOR REWRITE ---
                        score = results.get("score", 0)
                        max_score = results.get("max_score", 4)
                        score_pct = (score / max_score) * 100.0 if max_score > 0 else 0.0
                        
                        log_session_to_supabase(
                            unit=st.session_state.active_unit,
                            subtopic=st.session_state.active_topic,
                            app_mode="rewrite",
                            score_pct=score_pct,
                            keywords_used=results.get("key_terms_used", []),
                            keywords_missed=results.get("missed_terms", []),
                            misconception_flag=1 if score_pct < 45.0 else 0
                        )
                        st.rerun()
                else:
                    st.warning("Please type a rewrite before submitting.")
        else:
            res = st.session_state.rewrite_results
            st.success(f"🎉 **Rewrite Graded! Score: {res.get('score', 0)} / {res.get('max_score', 4)}**")
            
            with st.expander("📌 View Question & Your Submission", expanded=True):
                st.markdown(f"**Question:** {r_data.get('question')}")
                st.markdown(f"**Original Informal Draft:** *\"{r_data.get('layman_answer')}\"*")
                st.markdown(f"**Your Upgraded Rewrite:**\n\n> {st.session_state.student_rewrite_submission}")
            
            st.write("")
            st.markdown(f"**Technical Terms Added:** {', '.join(res.get('key_terms_used', [])) or 'None'}")
            st.markdown(f"**Missed Terms:** {', '.join(res.get('missed_terms', [])) or 'None'}")
            st.info(f"💡 **Examiner Note:** {res.get('feedback', '')}")
            
            st.write("")
            if st.button("🔄 Try Another Rewrite / Topic", type="primary", use_container_width=True):
                reset_session()