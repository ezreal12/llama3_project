
from datetime import datetime
import shutil
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Define LLM
llama3 = ChatOllama(model="llama3", temperature=0)
llama3_json = ChatOllama(model="llama3", format='json', temperature=0)

# Prompt Template
generate_prompt = PromptTemplate(
    template="Question: {question}\nAnswer:",
    input_variables=["question"]
)
generate_chain = generate_prompt | llama3 | StrOutputParser()

# Router
router_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>

    <|start_header_id|>system<|end_header_id|>

    You are an expert in determining the appropriate action based on the user's question. 
    If the question requires checking the computer's remaining storage capacity, return the keyword 'get_disc'.
    If the question asks for the current date and time, return the keyword 'get_date_time'.
    For all other cases that require a general response, return the keyword 'generate'.
    Provide the result in a JSON format with a single key 'choice' and no preamble or explanation.

    Question to evaluate: {question}

    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["question"],
)

# Chain
question_router = router_prompt | llama3_json | JsonOutputParser()

# State Definition
class GraphState(TypedDict):
    question: str
    generation: str
    datetime_info: str
    disk_info: str

# Router Node (always route to generate)
def route_question(state):
    """
    route question to web search or generation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("Step: Routing Query")
    question = state['question']
    output = question_router.invoke({"question": question})
    if output['choice'] == "get_disc":
        print("Step: get_disc")
        return "get_disc"
    elif output['choice'] == 'get_date_time':
        print("Step: get_date_time")
        return "get_date_time"
    elif output['choice'] == 'generate':
        print("Step: generate")
        return "generate"

# Generate Node
def generate(state):
    print("Generating response...")
    question = state["question"]
    generation = generate_chain.invoke({"question": question})
    return {"generation": generation}

# Get Date and Time Node
def get_date_time(state):
    print("Step: Retrieving current date and time")
    now = datetime.now()
    datetime_info = now.strftime("%Y-%m-%d %H:%M:%S")
    return {"datetime_info": datetime_info}

# Get Disk Space Node
def get_disc(state):
    print("Step: Checking disk space")
    total, used, free = shutil.disk_usage("/")
    disk_info = f"Free space: {free // (2**30)} GB"
    return {"disk_info": disk_info}



# Workflow Configuration
workflow = StateGraph(GraphState)
workflow.add_node("get_date_time", get_date_time)
workflow.add_node("get_disc", get_disc)
workflow.add_node("generate", generate)
workflow.set_conditional_entry_point(
    route_question,
    {
        "get_date_time": "get_date_time",
        "get_disc": "get_disc",
        "generate": "generate",
    },
)
workflow.add_edge("get_date_time", END)
workflow.add_edge("get_disc", END)
workflow.add_edge("generate", END)

local_agent = workflow.compile()

# Execution Function
def run_agent(query):
    output = local_agent.invoke({"question": query})
    print("Output:")
    print(output)

# Test
run_agent("What is the current time?")
run_agent("How much disk space is available?")
run_agent("Explain the basics of Hello World program.")