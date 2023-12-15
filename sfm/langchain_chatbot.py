#https://www.pragnakalp.com/create-slack-bot-using-python-tutorial-with-examples/
#test.py -> 터미널 하나더 ngrok http 5000 -> slack event subscription url 바꾸기
#https://api.slack.com/apps/A06A3P8H4EA/event-subscriptions?
import slack
from flask import Flask
from slackeventsapi import SlackEventAdapter

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores import FAISS

import os

SLACK_TOKEN="xoxb-6157885647652-6363066331953-PXAsCaUR4505uM0O6vEBqtE2"
SIGNING_SECRET="5dac6e18c85d42931f5747a81377120f"
 
app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)
 
client = slack.WebClient(token=SLACK_TOKEN)
##################################################################
os.environ["OPENAI_API_KEY"] = "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"
index_base_dir = './faiss_index'

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002' )
vectordb = FAISS.load_local(index_base_dir, embeddings)

#Question and answering
chatbot_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(
        temperature = 0.5, model_name = 'gpt-3.5-turbo', max_tokens = 500
    ),
    chain_type = "stuff",
    retriever = vectordb.as_retriever(search_kwargs={"k" : 2})
)

template = """
{query}? Check page_content of Document whether it contains reliable contents.
You also have to check source from metadata whether it can help you answer.
If you have no any information, just answer you don't have any related info.
Answer in Korean.
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template = template,
)


 
# @ slack_event_adapter.on('message')
# def message(payload):
#     # print(payload)
#     event = payload.get('event', {})
#     channel_id = event.get('channel')
#     user_id = event.get('user')
#     text = event.get('text')
#     print(user_id, text)
#     text_tmp = text
#     # if text == "hi":
#     #     print("got it!")
#     #     client.chat_postMessage(channel=channel_id,text="Hello")
#     # if text != answer:
#     if user_id != 'U06AP1Y9RU1':
#         answer = chatbot_chain.run(prompt.format(query = text))
#         client.chat_postMessage(channel=channel_id,text = answer )    
#     # else :
#     #     answer = ''
answer_dict = {}  # Dictionary to store the last answer for each user

@slack_event_adapter.on('message')
def message(payload):
    print(payload)
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')

    # Check if the user has asked a new question and it is not from the bot itself
    if text and text != answer_dict.get(user_id) and event.get("subtype") is None and user_id != "U06AP1Y9RU1":
        answer_dict[user_id] = text  # Update the last question for the user
        print('ans')

        # Rest of your logic for processing the question and generating an answer
        answer = chatbot_chain.run(prompt.format(query=text))
        client.chat_postMessage(channel=channel_id, text=answer)
    else:
        print("Ignoring subsequent messages from the same user or bot.")

        

##########################################################################################


# query = "공지사항을 알려주세요."
# print(chatbot_chain.run(prompt.format(query = query)))



if __name__ == "__main__":
    app.run(debug=True)