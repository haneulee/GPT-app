import json
import time
import os
import streamlit as st
from openai import OpenAI, AssistantEventHandler
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from bs4 import BeautifulSoup
import requests
from typing_extensions import override

ASSISTANT_NAME = "Research Assistant"

st.set_page_config(page_title="Research Assistant", layout="wide")
st.sidebar.title("Settings")

client = OpenAI(
    organization="org-l3stFe2ffnZM5sk17fbK0qKC",
    project="proj_X6ZoFeBCA6Dd4LssFgCguz8x",
)

# 사용자 OpenAI API 키 입력
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    client.beta.api_key = api_key
else:
    st.warning("Please enter your OpenAI API key to use the assistant.")
    st.stop()

st.sidebar.markdown(
    "[View on GitHub](https://github.com/haneulee/GPT-app/blob/main/pages/OpenAIAgent.py)"
)
st.title("🕵️ Research Assistant using OpenAI Assistant API")


class EventHandler(AssistantEventHandler):

    message = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message.replace("$", "\$"))

    def on_event(self, event):

        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)


# 도구 정의
def search_wikipedia(inputs):
    query = inputs["query"]
    url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs[:3])
    return "No Wikipedia page found."


def search_duckduckgo(inputs):
    try:
        ddg = DuckDuckGoSearchAPIWrapper()
        results = ddg.run(inputs["query"])
        return results if results else "No results found."
    except Exception as e:
        return f"Error fetching search results: {str(e)}"


def scrape_website(inputs):
    url = inputs["url"]
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))[:2000]
    return "Failed to scrape website."


def save_to_file(inputs):
    content = inputs["content"]
    filename = "research_result.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    return f"Results saved to {filename}"


functions_map = {
    "search_wikipedia": search_wikipedia,
    "search_duckduckgo": search_duckduckgo,
    "scrape_website": scrape_website,
    "save_to_file": save_to_file,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "Search DuckDuckGo for web results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Scrape a website's text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_file",
            "description": "Save research results to a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                    }
                },
                "required": ["content"],
            },
        },
    },
]


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with client.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


if "assistant" not in st.session_state:
    assistants = client.beta.assistants.list(limit=10)
    for a in assistants:
        if a.name == ASSISTANT_NAME:
            assistant = client.beta.assistants.retrieve(a.id)
            break
    else:
        assistant = client.beta.assistants.create(
            name=ASSISTANT_NAME,
            instructions="You help users do research on the given query using search engines. You give users the summarization of the information you got.",
            model="gpt-4o",
            tools=functions,
        )

    thread = client.beta.threads.create()
    st.session_state["assistant"] = assistant
    st.session_state["thread"] = thread
else:
    assistant = st.session_state["assistant"]
    thread = st.session_state["thread"]

paint_history(thread.id)
content = st.chat_input("What do you want to search?")
if content:
    send_message(thread.id, content)
    insert_message(content, "user")

    with st.chat_message("assistant"):
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
