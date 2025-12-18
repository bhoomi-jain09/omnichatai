from langgraph.graph import StateGraph,START,END
from langgraph.graph import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage
from typing import TypedDict,Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()
model=ChatGroq(model="llama-3.1-8b-instant")
class chatstate(TypedDict):
    message:Annotated[list[BaseMessage],add_messages]   #admessage is reducer add messages in list it doesnot overwrite

def chat_node(state:chatstate):
    #user query
    message = state.get("message", [])

    # prevent empty message → Groq 400 error
    if not message:
        return {"message": [HumanMessage(content="Hello! How can I help you?")]}

    result = model.invoke(message)
    return {"message": [result]}
#checkpointer=MemorySaver()
#yha inmemorysaver memory save on ram once the page refreash or restatrted the memory will crash so not relevent
conn=sqlite3.connect(database="chatbot_db",check_same_thread=False)
#sqllite3 use to connect with db if db not present it make new one 
# check_same_thread=false bcoz sqllite work on single thread only it check if the thread it make and use is same and we want to work on multiple thread so it doesnt check with false
 
checkpointer=SqliteSaver(conn=conn)
graph=StateGraph(chatstate)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
workflow=graph.compile(checkpointer=checkpointer)
def retrieve_all_threads():
    all_thread=set()
    for checkpoints in checkpointer.list(None):
        all_thread.add(checkpoints.config["configurable"]['thread_id'])
    return list(all_thread)

'''CONFIG = {'configurable': {'thread_id':'thread-2'}}
response=workflow.invoke(  
    {'message':[HumanMessage(content="my girlfriend name is bhoomi")]},
    config=CONFIG
)
print(response)'''