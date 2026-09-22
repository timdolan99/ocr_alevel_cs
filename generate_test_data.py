import os
import random
import json
from datetime import datetime, timedelta
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# --- 1. Database Connection ---
try:
    DB_URL = st.secrets["postgres"]["url"]
except Exception:
    DB_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:password@localhost:5432/postgres")

engine = create_engine(DB_URL, echo=False)
Base = declarative_base()

# --- 2. Matching Science ORM Schema ---
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

APP_MODES = ["socratic", "quiz", "extended", "rewrite"]

# --- 3. CS Keyword Map ---
CS_KEYWORD_MAP = {
    "1.1.1 Structure and function of the processor": {
        "keywords": ["ALU", "Control Unit", "Registers", "PC", "MAR", "MDR", "CIR", "ACC", "Buses", "FDE Cycle", "Pipelining"],
    },
    "1.2.1 Systems Software": {
        "keywords": ["OS", "Memory Management", "Paging", "Segmentation", "Virtual Memory", "Interrupts", "ISR", "Scheduling"],
    },
    "1.3.1 Compression, Encryption and Hashing": {
        "keywords": ["Lossy", "Lossless", "Run Length Encoding", "Dictionary Coding", "Symmetric Encryption", "Asymmetric Encryption"],
    },
    "1.4.2 Data Structures": {
        "keywords": ["Array", "Linked List", "Stack", "Queue", "Binary Search Tree", "Graph", "Hash Table", "Pointers"],
    },
    "2.1.1 Thinking abstractly": {
        "keywords": ["Abstraction", "Reality Modelling", "Information Hiding", "Simplification"],
    },
    "2.2.1 Programming techniques": {
        "keywords": ["Recursion", "Call Stack", "Base Case", "Scope", "By Value", "By Reference", "OOP", "Encapsulation"],
    },
    "2.3.1 Algorithms": {
        "keywords": ["Big-O Complexity", "Bubble Sort", "Insertion Sort", "Merge Sort", "Quick Sort", "Binary Search", "Dijkstra"],
    }
}

DEFAULT_KEYWORDS = ["Specification Concepts", "Syntax Mechanics", "Execution Logic", "Trace Evaluation"]

# --- 4. Generation Logic ---
def generate_synthetic_data(num_records=200):
    Session = sessionmaker(bind=engine)
    session = Session()

    spec_path = "course_spec.json"
    if os.path.exists(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            course_spec = json.load(f)
    else:
        print("Error: course_spec.json not found.")
        return

    topics = course_spec.get("topics", {})
    if not topics:
        print("Warning: course_spec.json has no topics.")
        return

    subject_title = course_spec.get("course_title", "OCR A-Level Computer Science")
    course_level = course_spec.get("level", "A-Level")

    print(f"Seeding {num_records} records matching the exact Science DDL schema...")

    start_date = datetime.now() - timedelta(days=30)

    for _ in range(num_records):
        unit_name = random.choice(list(topics.keys()))
        subtopic_name = random.choice(topics[unit_name])

        domain_meta = CS_KEYWORD_MAP.get(subtopic_name, {"keywords": DEFAULT_KEYWORDS})
        app_mode = random.choice(APP_MODES)

        base_score = random.choices([40.0, 55.0, 70.0, 85.0, 100.0], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]
        score_pct = max(10.0, min(100.0, base_score + random.uniform(-10.0, 10.0)))

        all_kw = list(domain_meta["keywords"])
        num_used = int((score_pct / 100.0) * len(all_kw))
        random.shuffle(all_kw)
        used_kw = all_kw[:num_used]
        missed_kw = all_kw[num_used:]

        # Integer flag: 1 if severe misconception, 0 otherwise
        misconception_flag = 1 if score_pct < 45.0 else 0

        timestamp = start_date + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(8, 17),
            minutes=random.randint(0, 59)
        )

        log = ActivityLog(
            timestamp=timestamp,
            subject=subject_title,
            level=course_level,
            unit=unit_name,
            subtopic=subtopic_name,
            app_mode=app_mode,
            score_pct=round(score_pct, 1),
            keywords_used=", ".join(used_kw) if used_kw else None,
            keywords_missed=", ".join(missed_kw) if missed_kw else None,
            misconception_flag=misconception_flag
        )

        session.add(log)

    session.commit()
    session.close()
    print(f"Successfully populated {num_records} activity logs!")

if __name__ == "__main__":
    generate_synthetic_data(1800)