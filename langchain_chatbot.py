from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores import FAISS

import os

os.environ["OPENAI_API_KEY"] = "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"
index_base_dir = './faiss_index'

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002' )
vectordb = FAISS.load_local(index_base_dir, embeddings)

#Question and answering
template = """
{query} page_content에서 필요한 정보를 찾으세요.
metadata 정보를 가지고 보낸 사람과 출처를 활용하세요. 해당 정보를 출력할 필요는 없습니다.
관련 정보를 자세하게 알려주세요.
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template = template,
)

chatbot_chain = RetrievalQA.from_chain_type(
		llm = ChatOpenAI(
        temperature = 0.5, model_name = 'gpt-3.5-turbo', max_tokens = 2000
        ),
		retriever = db.as_retriever(search_kwargs={"k" : 3}),
		verbose=True,
        chain_type = "stuff",
		chain_type_kwargs={
		        'document_prompt': PromptTemplate(
		            input_variables=["page_content", "sender_name", "source", "Date"], 
		            template="내용:\n{page_content}\n보낸 사람:{sender_name}\n출처:{source}\n보낸 날짜:{date}"
		        )
		    },
		)

print('Enter e or exit to quit chatting...')
query = input('Question: ')
while query != 'e' or query != 'exit':
    answer = chatbot_chain.run(prompt.format(query = query))
    print(answer)
    query = input('Question: ')