import panel as pn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

pn.extension('chat', theme='dark')

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_response(contents, user, instance):
    input_text = f"Answer the question: {contents}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    yield response

text_input = pn.chat.ChatAreaInput(
    placeholder='Type your message here...',
    resizable=False,
    height=60
)

chat_bot = pn.chat.ChatInterface(callback=get_response, widgets=text_input)

chat_bot.send("Ask me anything!", user="Assistant", respond=False)

template = pn.template.FastListTemplate(
    title='AI Chatbot',
    theme='dark'
)

chat_card = pn.Card(
    chat_bot,
    title='Chat with AI'
)

centered_chat = pn.Row(
    chat_card,
    align='center'
)

template.main.append(centered_chat)

template.servable()
