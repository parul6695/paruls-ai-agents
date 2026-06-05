---
title: Parul's AI Resume
emoji: 💼
colorFrom: indigo
colorTo: violet
sdk: gradio
sdk_version: "6.14.0"
app_file: resume_app.py
pinned: false
license: mit
---

# Parul's AI Resume

An interactive resume chatbot powered by GPT. Ask about my career, AI engineering projects, skills, and experience — grounded in my summary and LinkedIn profile.

## Example questions

- What AI and agent projects have you built?
- Tell me about your experience at Expedia.
- What technologies do you work with?
- How can I get in touch with you?

## Secrets (Space Settings → Repository secrets)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API access |
| `PUSHOVER_TOKEN` | Optional — notifications when users connect or ask unanswered questions |
| `PUSHOVER_USER` | Optional — Pushover user key |

## Optional assets

Add `me/avatar.png` for a profile photo in the chat (assistant avatar). Without it, the default avatar is used.

## Local development

```bash
cd 1_digital_resume
uv run resume_app.py
```
