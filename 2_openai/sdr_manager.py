import os
import re
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    trace,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
)
from agents.items import ToolCallItem, ToolCallOutputItem

from email_gen_agent import apply_sender_details, deliver_email, email_gen_agent, sender_instructions_block

load_dotenv(override=True)

AGENT_LABELS = {
    "sales_agent1": "OpenAI — Professional",
    "sales_agent2": "OpenRouter — Humorous",
    "sales_agent3": "Gemini — Concise",
}

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f8fafc",
    block_title_text_weight="600",
)

CUSTOM_CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: auto !important;
}
.hero-header {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
}
.hero-header h1 {
    margin-bottom: 0.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.hero-header p {
    color: var(--body-text-color-subdued);
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.6;
}
.draft-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: var(--background-fill-secondary);
}
.draft-card h4 {
    margin: 0 0 0.5rem;
    font-size: 0.95rem;
}
"""


class InputGuardrailOutput(BaseModel):
    should_block: bool
    info: str


class SdrApp:
    def __init__(self) -> None:
        self.sales_manager = self._build_sales_manager()

    def _build_sales_manager(self) -> Agent:
        sales_manager_instructions = f"""
        You are a Sales Manager at Priya Printers. Your goal is to find the single best cold sales email using the sales_agent tools.

        {sender_instructions_block()}

        Follow these steps carefully:
        1. Generate Drafts: Use all sales_agent tools to generate different email drafts. Do not proceed until all drafts from given tools are ready.

        2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.

        3. Handoff for Formatting: Pass ONLY the winning email draft to the 'Email Manager' agent. The Email Manager will format it with a subject line and HTML body.

        Crucial Rules:
        - Call each sales_agent tool exactly once. Do not retry.
        - Immediately hand off the best draft to Email Manager — do not ask the user for confirmation.
        - You must use the sales agent tools to generate the drafts — do not write them yourself.
        - You must hand off exactly ONE email to the Email Manager — never more than one.
        - Do not send the email. Formatting only.
        - Do not add action items or tell the user to fill in placeholders — sender details are already provided.
        """

        guardrail_agent = Agent(
            name="Input abuse check",
            instructions="""
            You guard inputs to a cold sales email generator. Block abuse and spam; allow normal B2B outreach parameters.

            Block (should_block=true) when the input includes:
            - Personal phone numbers, or other direct contact details used to harass, dox, or spam individuals
            - Spam or phishing patterns: bulk blasts, deceptive content, unrelated mass marketing, credential harvesting
            - Private sensitive data: SSN, passwords, home addresses, financial account numbers, leaked internal data
            - Impersonation of a specific real person using private details unrelated to outreach

            Allow (should_block=false) for legitimate sales email requests including:
            - Job titles and roles: CEO, CTO, VP Sales, Head of Marketing, "Dear CEO", etc.
            - Company and brand names (e.g. Acme Corp, Google)
            - Tone or style preferences (Professional, Humorous, Concise, etc.)
            - Business/prospect email addresses used for B2B cold outreach (e.g. prospect@company.com)
            - Sender signature details (name, title, company email, phone) that are part of normal email footers
            - Generic cold-outreach scenarios for commercial sales

            If blocking, set info to a brief reason. If allowing, set info to an empty string.
            """,
            output_type=InputGuardrailOutput,
            model="gpt-4o-mini",
        )

        @input_guardrail
        async def guardrail_against_abuse(ctx, agent, agent_input):
            result = await Runner.run(guardrail_agent, agent_input, context=ctx.context)
            should_block = result.final_output.should_block
            return GuardrailFunctionOutput(
                output_info={"blocked_input": result.final_output},
                tripwire_triggered=should_block,
            )

        tools, handoffs = email_gen_agent()
        return Agent(
            name="Sales Manager",
            instructions=sales_manager_instructions,
            tools=tools,
            handoffs=handoffs,
            model="gpt-4o-mini",
            input_guardrails=[guardrail_against_abuse],
        )

    @staticmethod
    def _build_message(recipient: str, company: str, tone: str) -> str:
        return (
            f"Write a cold sales email addressed to {recipient} at {company}. "
            f"The desired tone is {tone}. "
            f"The email is from the Head of Business Development at Priya Printers. "
            f"{sender_instructions_block()}"
        )

    @staticmethod
    def _strip_action_items(text: str) -> str:
        if not text:
            return text
        return re.split(r"\n\s*(?:\*\*)?Action Items(?:\*\*)?:", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    @staticmethod
    def _parse_final_output(text: str) -> tuple[str, str]:
        if not text:
            return "", ""

        text = SdrApp._strip_action_items(text)
        subject = ""
        html_body = ""

        html_match = re.search(r"```html\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if html_match:
            html_body = html_match.group(1).strip()
        else:
            tag_match = re.search(r"(<(?:html|body|div|p|table)\b.*)", text, re.DOTALL | re.IGNORECASE)
            if tag_match:
                html_body = tag_match.group(1).strip()

        subject_match = re.search(
            r"(?:\*\*)?Subject(?:\*\*)?:\s*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if subject_match:
            subject = subject_match.group(1).strip().strip("`")

        if not subject:
            plain_subject = re.search(r"^Subject:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            if plain_subject:
                subject = plain_subject.group(1).strip()

        return subject, html_body

    @staticmethod
    def _finalize_email_payload(
        subject: str,
        html_body: str,
        recipient_email: str,
    ) -> dict[str, str]:
        subject = apply_sender_details(subject, recipient_email)
        html_body = apply_sender_details(html_body, recipient_email)
        return {"subject": subject.strip(), "html_body": html_body.strip()}

    @staticmethod
    def _format_final(
        outputs: dict[str, str],
        final_output: Any,
        recipient_email: str = "",
    ) -> tuple[str, dict[str, str]]:
        subject = outputs.get("subject_writer", "").strip()
        html_body = outputs.get("html_converter", "").strip()

        if not subject or not html_body:
            parsed_subject, parsed_html = SdrApp._parse_final_output(str(final_output or ""))
            subject = subject or parsed_subject
            html_body = html_body or parsed_html

        payload = SdrApp._finalize_email_payload(subject, html_body, recipient_email)
        subject = payload["subject"]
        html_body = payload["html_body"]

        if subject or html_body:
            display = f"**Subject:** {subject or '(not generated)'}\n\n**HTML body:**\n```html\n{html_body}\n```"
            return display, payload

        text = str(final_output).strip() if final_output else ""
        display = text or "_No final email generated._"
        return display, {"subject": "", "html_body": ""}

    @staticmethod
    def _extract_tool_outputs(result) -> dict[str, str]:
        call_names: dict[str, str] = {}
        outputs: dict[str, str] = {}

        for item in result.new_items:
            if isinstance(item, ToolCallItem):
                call_id = item.call_id
                name = item.tool_name
                if call_id and name:
                    call_names[call_id] = name
            elif isinstance(item, ToolCallOutputItem):
                call_id = item.call_id
                tool_name = call_names.get(call_id or "", "")
                if tool_name:
                    outputs[tool_name] = str(item.output)

        return outputs

    @staticmethod
    def _format_drafts(outputs: dict[str, str]) -> str:
        if not outputs:
            return "_No drafts captured._"

        sections = []
        for tool_name in ("sales_agent1", "sales_agent2", "sales_agent3"):
            if tool_name not in outputs:
                continue
            label = AGENT_LABELS.get(tool_name, tool_name)
            body = outputs[tool_name].strip()
            sections.append(
                f'<div class="draft-card"><h4>{label}</h4><pre style="white-space: pre-wrap; margin: 0;">{body}</pre></div>'
            )
        return "\n".join(sections) if sections else "_No drafts captured._"

    async def generate(
        self, recipient: str, company: str, tone: str, recipient_email: str
    ) -> tuple[str, str, str, dict[str, str]]:
        if not recipient.strip() or not company.strip():
            return (
                "_Enter a recipient and company to generate drafts._",
                "",
                "Recipient and company are required.",
                {"subject": "", "html_body": "", "to_email": recipient_email},
            )

        message = self._build_message(recipient.strip(), company.strip(), tone)

        try:
            with trace("SDR Email Generator"):
                result = await Runner.run(self.sales_manager, message)
        except InputGuardrailTripwireTriggered as e:
            blocked = e.guardrail_result.output.output_info.get("blocked_input")
            reason = getattr(blocked, "info", str(blocked))
            return (
                "_Blocked by guardrail._",
                "",
                f"Blocked: input flagged as abusive or spam ({reason})",
                {"subject": "", "html_body": "", "to_email": recipient_email},
            )

        outputs = self._extract_tool_outputs(result)
        drafts_md = self._format_drafts(outputs)
        final_md, email_payload = self._format_final(outputs, result.final_output, recipient_email)
        email_payload["to_email"] = recipient_email
        status = "Drafts generated. Review the final email, then click **Send** when ready."
        return drafts_md, final_md, status, email_payload

    def send(self, email_state: dict[str, str]) -> str:
        subject = (email_state or {}).get("subject", "").strip()
        html_body = (email_state or {}).get("html_body", "").strip()
        to_email = (email_state or {}).get("to_email", "").strip()

        if not subject or not html_body:
            missing = []
            if not subject:
                missing.append("subject")
            if not html_body:
                missing.append("body")
            return f"Generate an email first — {' and '.join(missing)} missing."
        if not to_email:
            return "Enter a recipient email address before sending."

        try:
            deliver_email(subject, html_body, to_email)
        except Exception as exc:
            return f"Send failed: {exc}"

        return f"Email sent to {to_email}."


def create_demo(app: SdrApp) -> gr.Blocks:
    with gr.Blocks(title="SDR Email Generator") as demo:
        gr.Markdown(
            """
            <div class="hero-header">
                <h1>SDR Email Generator</h1>
                <p>
                    Generate cold sales email drafts with multiple AI agents, pick the best one,
                    preview the formatted result, and send when you're ready.
                </p>
            </div>
            """,
        )

        email_state = gr.State({"subject": "", "html_body": "", "to_email": ""})

        with gr.Row():
            recipient = gr.Textbox(label="Recipient", placeholder="Dear CEO")
            company = gr.Textbox(label="Company", placeholder="Acme Corp")
            tone = gr.Dropdown(
                ["Professional", "Humorous", "Concise", "Friendly", "Direct"],
                label="Tone",
                value="Professional",
            )

        recipient_email = gr.Textbox(
            label="Recipient email",
            placeholder="prospect@company.com",
            info="Used when you click Send",
        )

        generate_btn = gr.Button("Generate emails", variant="primary")

        drafts_output = gr.Markdown(label="Drafts")
        final_output = gr.Markdown(label="Final email")
        status_output = gr.Textbox(label="Status", interactive=False)

        send_btn = gr.Button("Send email", variant="secondary")

        generate_btn.click(
            fn=app.generate,
            inputs=[recipient, company, tone, recipient_email],
            outputs=[drafts_output, final_output, status_output, email_state],
        )
        send_btn.click(
            fn=app.send,
            inputs=[email_state],
            outputs=[status_output],
        )

    return demo


if __name__ == "__main__":
    create_demo(SdrApp()).launch(theme=THEME, css=CUSTOM_CSS)
