import os
import json
from pyexpat.errors import messages
from click import argument
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader
import requests
from openai import OpenAI
load_dotenv(override=True)

def push(message):
    print(message)
    response=requests.post( 
    "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": message,
        })
    print("Pushover response: ", response.json())

def record_new_user(name="Name not provided",email="",notes="Notes not provided"):
    push(f"Recorded New user with name: {name} email {email}, {notes}")
    return {"New user recorded" : "ok"}

def record_new_questions(question=""):
    push(f"Recorded New question: {question} ")
    return {"New question recorded" : "ok"}

record_new_user_json={
    "name": "record_new_user",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string",
            "description": "The name of the user, if they provide it"},
            "email": {"type": "string",
            "description": "The email address of the user"},
            "notes": {"type": "string",
            "description": "Any additional information about the conversation that's worth recording to give context to the question"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_new_questions_json={
    "name": "record_new_questions",
    "description": "Use this tool to record user's question that you are unable to answer even after using summary and linkeind information provided",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string",
            "description": "The question that the user asked"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools=[{"type":"function", "function": record_new_user_json}, 
       {"type":"function", "function": record_new_questions_json}]

class ResumeApp:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Parul"
        self.linkedin=""
        self.reader = PdfReader("me/linkedin.pdf")
        for page in self.reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        
        self.summary=""
        with open("me/summary.txt", "r") as f:
            self.summary = f.read()
    
    # few notes 
    # tool_call.function.arguments is a string containing JSON of args, not a dict.
    # Parse = json.loads(...) ,json.loads reads that string and builds a Python dict
    # That dict is stored in arguments.
    # **arguments when calling the function
    # spreads the dict into keyword arguments for the function
    # The tool schema you send to the model (record_user_details_json, etc.) 
    # tells it which fields exist (email, name, question, …). 
    # The model fills those in and the API returns them as JSON text. 
    # Parsing is required because Python can’t call record_user_details(email=...) until that text is a real dict.
    # So in short: parsed JSON args = the tool’s argument string converted from JSON into a Python dict, 
    # then passed into your tool function as keyword arguments.
    # json.dumps(result) turns the Python value returned by your tool into a JSON string for the content field of a tool message
    # Chat Completions API expects each tool message’s content to be a string, not a Python dict. 
    # json.dumps(result) serializes that dict to something like:
    # "{\"recorded\": \"ok\"}"
    # Arguments arrive as a JSON string from the API; 
    # you decode them, run the function, then encode the return value back to JSON for the next turn.

    def handle_tool_calls(self, tool_calls):
        results=[]
        for tool_call in tool_calls:
            print(f"tool called : {tool_call}", flush=True )
            tool_name=tool_call.function.name
            tool = globals().get(tool_name)
            argument_str = tool_call.function.arguments
            arguments= json.loads(argument_str)
            result=tool(**arguments)
            results.append({"role":"tool", "content":json.dumps(result),"tool_call_id":tool_call.id})
        return results


    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_new_questions tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_new_user tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def chat(self, message, history):
        messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + normalize_history(history)
            + [{"role": "user", "content": message}]
        )
        done = False
        while not done:
            response= self.openai.chat.completions.create(model="gpt-4.1-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done=True
        return response.choices[0].message.content

    
APP_DIR = os.path.dirname(os.path.abspath(__file__))

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
    max-width: 920px !important;
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
"""

EXAMPLE_QUESTIONS = [
    "What technologies do you work with?",
    "How can I get in touch with you?",
]


def normalize_history(history):
    """Convert Gradio 6 message blocks to plain strings for the OpenAI API."""
    normalized = []
    for message in history:
        content = message["content"]
        if isinstance(content, list):
            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        normalized.append({"role": message["role"], "content": content})
    return normalized


def create_demo(resume: ResumeApp):
    avatar_path = os.path.join(APP_DIR, "me", "avatar.png")
    chatbot_kwargs = {"buttons": ["copy"]}
    if os.path.exists(avatar_path):
        chatbot_kwargs["avatar_images"] = (None, avatar_path)

    with gr.Blocks(title="Chat with Parul") as demo:
        gr.Markdown(
            f"""
            <div class="hero-header">
                <h1>Hi, I'm {resume.name} 👋</h1>
                <p>
                    Ask me about my background, AI engineering work, and experience building
                    production agent systems. Leave your email if you'd like to connect.
                </p>
            </div>
            """,
        )
        gr.ChatInterface(
            resume.chat,
            chatbot=gr.Chatbot(**chatbot_kwargs),
            textbox=gr.Textbox(placeholder="Ask about my career, skills, or projects…"),
            examples=EXAMPLE_QUESTIONS,
            fill_height=True,
        )
    return demo


if __name__ == "__main__":
    resume = ResumeApp()
    create_demo(resume).launch(theme=THEME, css=CUSTOM_CSS)

