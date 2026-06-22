import os
from typing import Dict

import sendgrid
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, function_tool
from sendgrid.helpers.mail import Content, Email, Mail, To

load_dotenv(override=True)

SENDER = {
    "name": os.getenv("SENDER_NAME", "Rahul Gupta"),
    "title": os.getenv("SENDER_TITLE", "Head of Business Development"),
    "company": os.getenv("SENDER_COMPANY", "Priya Printers"),
    "email": os.getenv("SENDER_EMAIL", "parulmscs@gmail.com"),
    "phone": os.getenv("SENDER_PHONE", ""),
    "website": "https://www.priyaprinters.com/",
}


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
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email(SENDER["email"])
    to = To(to_email)
    content = Content("text/html", html_body)
    mail = Mail(from_email, to, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
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
    {sender_block}
  """
    instructions2 = f"""You are a humorous, engaging sales agent working for Priya Printers. {company_blurb}
    You write witty, engaging cold emails that are likely to get a response.
    {sender_block}
  """
    instructions3 = f"""You are a busy sales agent working for Priya Printers. {company_blurb}
    You write concise, to the point cold emails.
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

    description = "Write a cold sales email"

    tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
    tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
    tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

    subject_instructions = "You can write a subject for a cold sales email. \
    You are given a message and you need to write a subject for an email that is likely to get a response."

    html_instructions = f"""You convert a text email body to an HTML email body.
    Use a simple, clear, compelling layout. Include a footer with the sender's real name, title, company, and contact details.
    Never use bracket placeholders — use the sender details below.
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
        handoff_description="Convert an email to HTML with a subject line",
    )
    
    #Add tool3 once you have paid subscription to gemini models
    # tools = [tool1,tool2,tool3]
    tools = [tool1,tool2]
    handoffs = [emailer_agent]
    return tools,handoffs



