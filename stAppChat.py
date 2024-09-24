import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Chatbot", layout="centered")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto").to("cuda")

if "history" not in st.session_state:
    st.session_state.history = []

def add_message(sender, message):
    st.session_state.history.append({"sender": sender, "message": message})

st.title("AI Chatbot")
st.write("Ask any question and AI will answer.")



user_input = st.text_input("Your message:")


if st.button("Send"):
    if user_input:
        add_message("user", user_input)

        input_text = f"answer the question: {user_input}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        add_message("assistant", response)

if st.button("Clear chat"):
    st.session_state.history = []

st.write("### Chat:")
chat_container = st.container()
with chat_container:
    chat_history_css = """
        <style>
        .chat-history {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 20px;
            scrollbar-width: thin;
            scrollbar-color: #555 #33372C;
        }

        .chat-history::-webkit-scrollbar {
            width: 8px;
        }

        .chat-history::-webkit-scrollbar-track {
            background: #33372C;
        }

        .chat-history::-webkit-scrollbar-thumb {
            background-color: #555;
            border-radius: 10px;
            border: 2px solid #33372C;
        }

        .chat-history::-webkit-scrollbar-thumb:hover {
            background-color: #777;
        }

        .chat-history::-webkit-scrollbar-button {
            display: none;
        }
        </style>
    """
    chat_content = "<div class='chat-history'>"
    for chat in st.session_state.history:
        if chat["sender"] == "user":
            chat_content += f"""
            <div style='font-family: "Roboto", sans-serif; font-style: normal; text-align: right; background-color: #654520; color: white; padding: 10px; border-radius: 10px; margin: 5px;'>
                <strong>Ty:</strong> {chat['message']}
            </div>
            """
        else:
            chat_content += f"""
            <div style='font-family: "Roboto", sans-serif; font-style: normal; text-align: left; background-color: #33372C; color: white; padding: 10px; border-radius: 10px; margin: 5px;'>
                <strong>Asystent:</strong> {chat['message']}
            </div>
            """
    chat_content += "</div>"

    scroll_js = """
        <script>
        var chatDiv = document.getElementsByClassName('chat-history')[0];
        chatDiv.scrollTop = chatDiv.scrollHeight;
        </script>
    """

    components.html(chat_history_css + chat_content + scroll_js, height=400)
