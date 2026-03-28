import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Core helper: call the model ----------
def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    """
    Calls an LLM with a system prompt (instructions) and user prompt (inputs).
    Lower temperature => more consistent outputs.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------- Utility: simple template fill ----------
def fill_template(template: str, **kwargs) -> str:
    for k, v in kwargs.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    return template


# =========================
# 1) EMAIL SUMMARIZATION
# =========================

EMAIL_SUMMARY_SYSTEM = """
You are an executive assistant specializing in concise, accurate communication.
Your job is to summarize emails for busy professionals.
"""

EMAIL_SUMMARY_TEMPLATE = """
TASK:
Summarize the email below for a busy professional.

RULES:
- Be factual and do not invent details.
- Keep the summary short and actionable.
- Capture key requests, deadlines, and next steps.
- If the email has questions, list them.

OUTPUT FORMAT (exactly):
Summary: <2-4 sentences>
Key Points:
- <bullet>
- <bullet>
Action Items:
- <bullet>
Questions:
- <bullet or 'None'>
Tone: <Neutral / Urgent / Friendly / Negative>

EMAIL:
{{email_text}}
"""

def summarize_email(email_text: str) -> str:
    user_prompt = fill_template(EMAIL_SUMMARY_TEMPLATE, email_text=email_text)
    return call_llm(EMAIL_SUMMARY_SYSTEM, user_prompt)


# =========================
# 2) RESUME SCREENING
# =========================

RESUME_SCREEN_SYSTEM = """
You are a technical recruiter screening candidates for an AI Engineer role.
Be strict, evidence-based, and consistent.
Only use information provided in the resume.
"""

RESUME_SCREEN_TEMPLATE = """
TASK:
Evaluate the resume against the job requirements and produce a hiring recommendation.

JOB REQUIREMENTS:
{{job_requirements}}

RESUME:
{{resume_text}}

RULES:
- Do not assume missing info.
- If a requirement is not explicitly met, mark it as "Not evidenced".
- Cite evidence using short quotes or bullet references from the resume.
- Be concise and structured.

OUTPUT FORMAT (exactly):
Recommendation: <Strong Yes / Yes / Maybe / No>
Score (0-100): <number>
Top Strengths:
- <bullet>
Top Risks / Gaps:
- <bullet>
Requirements Match:
- Requirement: <Met / Partial / Not evidenced> — Evidence: <text>
- Requirement: <Met / Partial / Not evidenced> — Evidence: <text>
Interview Questions to Validate:
- <bullet>
"""

def screen_resume(resume_text: str, job_requirements: str) -> str:
    user_prompt = fill_template(
        RESUME_SCREEN_TEMPLATE,
        resume_text=resume_text,
        job_requirements=job_requirements
    )
    return call_llm(RESUME_SCREEN_SYSTEM, user_prompt, temperature=0.1)


# =========================
# 3) CUSTOMER SUPPORT REPLY
# =========================

SUPPORT_REPLY_SYSTEM = """
You are a customer support agent for a software product.
You are empathetic, accurate, and solution-oriented.
You must follow the company policy.
"""

SUPPORT_REPLY_TEMPLATE = """
TASK:
Write a high-quality customer support response.

PRODUCT CONTEXT:
{{product_context}}

COMPANY POLICY:
{{policy_text}}

CUSTOMER MESSAGE:
{{customer_message}}

RULES:
- Do not promise refunds or actions that violate policy.
- If info is missing, ask 1-3 clarifying questions.
- Use a warm but professional tone.
- Provide step-by-step troubleshooting if relevant.
- End with a clear next step.

OUTPUT FORMAT:
Subject: <short subject line>
Response:
<email-style response>
Clarifying Questions (if any):
- <question or 'None'>
"""

def generate_support_reply(customer_message: str, product_context: str, policy_text: str) -> str:
    user_prompt = fill_template(
        SUPPORT_REPLY_TEMPLATE,
        customer_message=customer_message,
        product_context=product_context,
        policy_text=policy_text
    )
    return call_llm(SUPPORT_REPLY_SYSTEM, user_prompt, temperature=0.3)


# =========================
# DEMO RUN
# =========================
if __name__ == "__main__":
    # 1) Email summarization demo
    sample_email = """
    Subject: Q1 Planning - Need your input by Friday

    Hi Vivian,
    Can you review the attached Q1 plan and send feedback by Friday 3pm ET?
    Specifically, we need your thoughts on the budget line items and the hiring timeline.
    Also, are you able to present the final plan in Monday’s leadership meeting?

    Thanks,
    Jordan
    """
    print("\n=== EMAIL SUMMARY ===")
    print(summarize_email(sample_email))

    # 2) Resume screening demo
    sample_job_req = """
    - 3+ years Python engineering
    - Experience with LLM APIs (OpenAI, Anthropic, etc.)
    - Experience building RAG pipelines (embeddings + vector DB)
    - FastAPI or similar for deployment
    - Familiarity with cloud (AWS/Azure/GCP)
    """
    sample_resume = """
    Jane Doe
    Experience:
    - Software Engineer (3.5 years): Built Python services, REST APIs, and internal tools.
    - AI Projects: Integrated OpenAI API for summarization and classification.
    - Built a document Q&A prototype using embeddings and FAISS.
    - Deployed a small FastAPI service for internal usage.
    Skills: Python, FastAPI, Docker, OpenAI API, FAISS, Git, AWS (EC2 basics)
    """
    print("\n=== RESUME SCREEN ===")
    print(screen_resume(sample_resume, sample_job_req))

    # 3) Support reply demo
    product_context = """
    Product: School of AI Bootcamp Portal
    Features: login, course videos, community access, certificate download
    Common issues: password reset email not received, video playback errors, access not granted
    """
    policy_text = """
    - Refunds are not handled by support; direct users to billing team.
    - Support can reset access, resend verification, and troubleshoot login/video issues.
    - Never ask for full passwords; only request email and last 4 digits of order ID if needed.
    """
    customer_msg = "Hi, I signed up but I can’t access the bootcamp videos. It says I don’t have permission."
    print("\n=== SUPPORT REPLY ===")
    print(generate_support_reply(customer_msg, product_context, policy_text))
