import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Setup
# =========================
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# Core LLM Call
# =========================
def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"


# =========================
# Utility: Template Fill
# =========================
def fill_template(template: str, **kwargs) -> str:
    for k, v in kwargs.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    return template


# =========================
# 1) Email Summarizer
# =========================
EMAIL_SYSTEM = "You are a professional assistant that summarizes emails."

EMAIL_TEMPLATE = """
Summarize the email below.

Rules:
- Be concise
- Capture key request
- Mention deadline if present

Output Format:
Summary:
Action Items:
Deadline:

Email:
{{email_text}}
"""

def summarize_email(email_text: str) -> str:
    prompt = fill_template(EMAIL_TEMPLATE, email_text=email_text)
    return call_llm(EMAIL_SYSTEM, prompt)


# =========================
# 2) Resume Summarizer
# =========================
RESUME_SYSTEM = "You are an HR assistant specializing in resume analysis."

RESUME_TEMPLATE = """
Summarize the resume below.

Output Format:
Summary:
Key Skills:
Experience Level:
Best Fit Roles:

Resume:
{{resume_text}}
"""

def summarize_resume(resume_text: str) -> str:
    prompt = fill_template(RESUME_TEMPLATE, resume_text=resume_text)
    return call_llm(RESUME_SYSTEM, prompt)


# =========================
# 3) Resume Screening
# =========================
SCREEN_SYSTEM = "You are a recruiter screening candidates."

SCREEN_TEMPLATE = """
Evaluate the resume against job requirements.

Output Format:
Recommendation:
Score:
Strengths:
Gaps:
Hiring Risk:

Job Requirements:
{{job_requirements}}

Resume:
{{resume_text}}
"""

def screen_resume(resume_text: str, job_requirements: str) -> str:
    prompt = fill_template(
        SCREEN_TEMPLATE,
        resume_text=resume_text,
        job_requirements=job_requirements
    )
    return call_llm(SCREEN_SYSTEM, prompt, temperature=0.1)


# =========================
# 4) Interview Questions
# =========================
INTERVIEW_SYSTEM = "You are a senior interviewer."

INTERVIEW_TEMPLATE = """
Create interview questions.

Output:
Technical Questions:
Behavioral Questions:
Scenario Questions:

Role:
{{job_role}}
"""

def generate_interview_questions(job_role: str) -> str:
    prompt = fill_template(INTERVIEW_TEMPLATE, job_role=job_role)
    return call_llm(INTERVIEW_SYSTEM, prompt)


# =========================
# 5) Support Reply Generator
# =========================
SUPPORT_SYSTEM = "You are a helpful customer support assistant."

SUPPORT_TEMPLATE = """
Write a helpful reply.

Customer Message:
{{customer_msg}}

Product Context:
{{product_context}}

Policy:
{{policy_text}}
"""

def generate_support_reply(customer_msg, product_context, policy_text):
    prompt = fill_template(
        SUPPORT_TEMPLATE,
        customer_msg=customer_msg,
        product_context=product_context,
        policy_text=policy_text
    )
    return call_llm(SUPPORT_SYSTEM, prompt)


# =========================
# DEMO RUN
# =========================
if __name__ == "__main__":

    # 1) Email Summary
    email = """
    Subject: Q1 Planning

    Hi Vivian,
    Please review the Q1 plan and send feedback by Friday 3pm.
    Also confirm if you can present Monday.

    Thanks,
    Jordan
    """
    print("\n=== EMAIL SUMMARY ===")
    print(summarize_email(email))


    # 2) Resume Screening
    job_req = """
    - Python
    - LLM APIs
    - RAG pipelines
    - FastAPI
    """
    resume = """
    Software Engineer with 3.5 years experience.
    Built APIs, used OpenAI, created RAG system with FAISS.
    """
    print("\n=== RESUME SCREEN ===")
    print(screen_resume(resume, job_req))


    # 3) Interview Questions
    print("\n=== INTERVIEW QUESTIONS ===")
    print(generate_interview_questions("AI Engineer"))


    # 4) Support Reply
    product_context = "AI Bootcamp platform with videos and login"
    policy = "Support can help with login and access issues"
    customer_msg = "I cannot access my course"

    print("\n=== SUPPORT REPLY ===")
    print(generate_support_reply(customer_msg, product_context, policy))