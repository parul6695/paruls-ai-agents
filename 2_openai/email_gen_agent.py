import os
import re
import warnings
from typing import Dict

import sendgrid
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, function_tool
from sendgrid.helpers.mail import Content, Email, Mail, ReplyTo, To

load_dotenv(override=True)

FREE_EMAIL_DOMAINS = frozenset(
    {"gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com"}
)

SENDER = {
    "name": os.getenv("SENDER_NAME", "Priya Printers"),
    "title": os.getenv("SENDER_TITLE", "Head of Business Development"),
    "company": os.getenv("SENDER_COMPANY", "Priya Printers"),
    "email": os.getenv("SENDER_EMAIL", "sales@priyaprinters.com"),
    "phone": os.getenv("SENDER_PHONE", ""),
    "website": "https://www.priyaprinters.com/",
}

# Must be a SendGrid-verified sender on your own domain (not gmail/yahoo).
# Using a free-provider address here is the main reason Gmail shows "via sendgrid.net" and files mail as spam.
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", SENDER["email"])

ANTI_SPAM_SUBJECT_RULES = """
Write a short, plain subject line (under 60 characters) that sounds like a real person wrote it.
- Use sentence case, not Title Case Every Word
- No exclamation marks, no ALL CAPS, no emojis
- Avoid spam phrases: "Elevate", "Unlock", "Transform", "Revolutionize", "Don't miss", "Limited time", "Act now", "Custom Solutions", "Boost your"
- Prefer specific and understated: "packaging question for {company}", "print partner for {company}", "quick note — Priya Printers"
"""

ANTI_SPAM_BODY_RULES = """
Write like one person emailing another — not a marketing blast.
- 3–5 short paragraphs max; plain language, no hype or superlatives
- One clear, low-pressure ask (e.g. open to a 10-minute call?)
- No bullet lists of product features unless the recipient asked
- End with a real signature using the sender details provided
- Include one line: "If this isn't relevant, just reply and I won't follow up."
"""


def sender_instructions_block() -> str:
    contact = SENDER["email"]
    if SENDER["phone"]:
        contact = f"{SENDER['email']} | {SENDER['phone']}"
    return f"""
Sender details — use these exact values in every email signature and footer.
Never use bracket placeholders like [Your Name] or [Your Contact Information].
- Name: {SENDER['name']}
- Title: {SENDER['title']}
- Company: {SENDER['company']}
- Contact: {contact}
- Website: {SENDER['website']}
"""


def clean_subject(subject: str) -> str:
    if not subject:
        return subject
    cleaned = subject.strip().strip("`")
    cleaned = re.sub(r"^(?:\*\*)?Subject(?:\*\*)?:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"!+$", "", cleaned).strip()
    cleaned = re.sub(
        r"\b(elevate|unlock|transform|revolutionize|boost|supercharge)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -—")
    if cleaned and cleaned == cleaned.upper() and len(cleaned) > 4:
        cleaned = cleaned.capitalize()
    return cleaned.strip()


def _from_domain(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower().strip()


def _warn_if_risky_from_address() -> None:
    domain = _from_domain(SENDGRID_FROM_EMAIL)
    if domain in FREE_EMAIL_DOMAINS:
        warnings.warn(
            f"SENDGRID_FROM_EMAIL is {SENDGRID_FROM_EMAIL!r}. Gmail/Yahoo addresses sent through "
            "SendGrid fail DMARC and usually land in spam (shown as 'via sendgrid.net'). "
            "Authenticate priyaprinters.com in SendGrid and set SENDGRID_FROM_EMAIL to e.g. "
            "sales@priyaprinters.com.",
            UserWarning,
            stacklevel=3,
        )


def clean_html_body(html_body: str) -> str:
    if not html_body:
        return html_body

    text = html_body.strip()
    fence_match = re.search(
        r"^(?:```|''')(?:html)?\s*\n?(.*?)(?:\n?(?:```|'''))?\s*$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    text = re.sub(r"^(?:```|''')html\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?(?:```|''')\s*$", "", text)

    tag_match = re.search(r"(<(?:html|body|div|p|table|!DOCTYPE)\b.*)", text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        text = tag_match.group(1).strip()

    return text.strip()


def html_to_plain_text(html_body: str) -> str:
    if not html_body:
        return html_body

    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_body, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def apply_sender_details(text: str, recipient_email: str = "") -> str:
    if not text:
        return text

    contact = SENDER["email"]
    if SENDER["phone"]:
        contact = f"{SENDER['email']} | {SENDER['phone']}"

    replacements = {
        "[Your Name]": SENDER["name"],
        "[Your Contact Information]": contact,
        "[Recipient's Email Address]": recipient_email,
        "[Recipient Email Address]": recipient_email,
    }
    for placeholder, value in replacements.items():
        if value:
            text = text.replace(placeholder, value)
    return text


def deliver_email(subject: str, html_body: str, to_email: str) -> Dict[str, str]:
    _warn_if_risky_from_address()

    subject = clean_subject(apply_sender_details(subject, to_email))
    html_body = clean_html_body(apply_sender_details(html_body, to_email))
    plain_body = html_to_plain_text(html_body)

    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email(SENDGRID_FROM_EMAIL, SENDER["name"])
    to = To(to_email)
    mail = Mail(from_email, to, subject)
    if SENDGRID_FROM_EMAIL.lower() != SENDER["email"].lower():
        mail.reply_to = ReplyTo(SENDER["email"], SENDER["name"])
    mail.add_content(Content("text/plain", plain_body))
    mail.add_content(Content("text/html", html_body))
    sg.client.mail.send.post(request_body=mail.get())
    return {"status": "success"}


@function_tool
def send_html_email(subject: str, html_body: str, to_email: str) -> Dict[str, str]:
    """Send an email with the given subject and HTML body to the recipient."""
    return deliver_email(subject, html_body, to_email)

def email_gen_agent():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    if openrouter_api_key:
        print(f"OPENROUTER API Key exists and begins {openrouter_api_key[:8]}")
    if gemini_api_key:
        print(f"GEMINI API Key exists and begins {gemini_api_key[:8]}")
    

    sender_block = sender_instructions_block()
    company_blurb = (
        "Priya Printers (https://www.priyaprinters.com/) is a Delhi-based manufacturing & printing partner "
        "delivering high-quality, customised packaging and commercial print for brands across India and overseas. "
        "Three decades of craft, modern production standards."
    )
    instructions1 = f"""You are a sales agent working for Priya Printers. {company_blurb}
    You write professional, serious cold emails.
    {ANTI_SPAM_BODY_RULES}
    {sender_block}
  """
    instructions2 = f"""You are a humorous, engaging sales agent working for Priya Printers. {company_blurb}
    You write witty, engaging cold emails that are likely to get a response — still human and understated, never gimmicky or salesy.
    {ANTI_SPAM_BODY_RULES}
    {sender_block}
  """
    instructions3 = f"""You are a busy sales agent working for Priya Printers. {company_blurb}
    You write concise, to the point cold emails.
    {ANTI_SPAM_BODY_RULES}
    {sender_block}
  """
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    openrouter_client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=openrouter_api_key,
        max_retries=5,
    )
    gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=gemini_api_key)

    openai_model = OpenAIChatCompletionsModel(model='gpt-4o',openai_client=openai_client)
    # openrouter/free auto-selects an available free model to avoid single-model rate limits
    openrouter_model = OpenAIChatCompletionsModel(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        openai_client=openrouter_client,
    )
    gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=gemini_client)

    sales_agent1 = Agent(name="Open AI Sales Agent", instructions=instructions1, model=openai_model)
    sales_agent2 = Agent(name="OpenRouter Sales Agent", instructions=instructions2, model=openrouter_model)
    sales_agent3 = Agent(name="Gemini Sales Agent", instructions=instructions3, model=gemini_model)

    sales_agents = {
        "sales_agent1": sales_agent1,
        "sales_agent2": sales_agent2,
        # Add sales_agent3 once you have a paid Gemini subscription.
        # "sales_agent3": sales_agent3,
    }

    subject_instructions = (
        "You write the subject line for a one-to-one B2B cold email. "
        "Return only the subject text — no label, quotes, or punctuation spam.\n"
        f"{ANTI_SPAM_SUBJECT_RULES}"
    )

    html_instructions = f"""You convert a text email body to an HTML email body.
    Use a simple, clear layout — no heavy marketing banners, gradient buttons, or stock-photo styling.
    Include a footer with the sender's real name, title, company, and contact details.
    Never use bracket placeholders — use the sender details below.
    Return raw HTML only. Do not wrap the output in markdown code fences like ```html or '''html.
    {sender_block}
    """

    subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model="gpt-4o-mini")
    subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email")

    html_converter = Agent(name="HTML email body converter", instructions=html_instructions, model="gpt-4o-mini")
    html_tool = html_converter.as_tool(tool_name="html_converter",tool_description="Convert a text email body to an HTML email body")

    email_tools = [subject_tool, html_tool, send_html_email]

    instructions = (
        "You are an email formatter. You receive the body of an email to be sent. "
        "First use the subject_writer tool to write a subject, "
        "then use the html_converter tool to convert the body to HTML. "
        "Return only the final subject line and HTML body. "
        "Do not send the email. Do not add action items, notes to the user, or placeholder brackets."
    )

    emailer_agent = Agent(
        name="Email Manager",
        instructions=instructions,
        tools=email_tools,
        model="gpt-4o-mini",
    )

    return sales_agents, emailer_agent



