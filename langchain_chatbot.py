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
# chatbot_chain = RetrievalQA.from_chain_type(
#     llm = ChatOpenAI(
#         temperature = 0.5, model_name = 'gpt-3.5-turbo', max_tokens = 500
#     ),
#     chain_type = "stuff",
#     retriever = vectordb.as_retriever(search_kwargs={"k" : 2})
# )
chatbot_chain = RetrievalQA.from_chain_type(
		llm = ChatOpenAI(
        temperature = 0.5, model_name = 'gpt-3.5-turbo', max_tokens = 500
        ),
		retriever = db.as_retriever(search_kwargs={"k" : 2}),
		verbose=True, 
		chain_type_kwargs={
		        'document_prompt': PromptTemplate(
		            input_variables=["page_content", "sender_name", "source", "Date"], 
		            template="Context:\n{page_content}\nSender:{sender_name}\nSource:{source}\nDate:{date}"
		        ),
		    },
		)

template = """
{query}. Check page_content of Document whether it contains reliable contents.
You also have to check source from metadata whether it can help you answer.
If you cannot find related information, just answer you don't have any related info.
Answer in Korean.
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template = template,
)

print('Enter e or exit to quit chatting...')
query = input('Question: ')
while query != 'e' or query != 'exit':
    answer = chatbot_chain.run(prompt.format(query = query))
    print(answer)
    query = input('Question: ')