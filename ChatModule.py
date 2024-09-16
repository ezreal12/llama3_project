# Displaying final output format
# pip install --upgrade --quiet duckduckgo-search
from IPython.display import display, Markdown, Latex
# LangChain Dependencies
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import END, StateGraph
# For State Graph 
from typing_extensions import TypedDict
import os


class ChatModule:
    def __init__(self):
        # Defining LLM
        # 중요: 딱히 llama3가 아니여도 동작함. 검색요약 결과가 다를뿐임
        #local_llm = 'llama2'
        #local_llm = 'llama3'
        local_llm = 'openhermes'
        self.llama3 = ChatOllama(model=local_llm, temperature=0)
        self.llama3_json = ChatOllama(model=local_llm, format='json', temperature=0)

        self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
        self.web_search_tool = DuckDuckGoSearchRun(api_wrapper=self.wrapper)

        # Generation Prompt
        self.generate_prompt = PromptTemplate(
            template="""
            
            <|begin_of_text|>
            
            <|start_header_id|>system<|end_header_id|> 
            
            You are an AI assistant for Research Question Tasks, that synthesizes web search results. 
            Strictly use the following pieces of web search context to answer the question. If you don't know the answer, just say that you don't know. 
            keep the answer concise, but provide all of the details you can in the form of a research report. 
            Only make direct references to material if provided in the context.
            
            <|eot_id|>
            
            <|start_header_id|>user<|end_header_id|>
            
            Question: {question} 
            Web Search Context: {context} 
            Answer: 
            
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "context"],
        )

        # Chain
        self.generate_chain = self.generate_prompt | self.llama3 | StrOutputParser()

        # Router
        self.router_prompt = PromptTemplate(
            template="""
            
            <|begin_of_text|>
            
            <|start_header_id|>system<|end_header_id|>
            
            You are an expert at routing a user question to either the generation stage or web search. 
            Use the web search for questions that require more context for a better answer, or recent events.
            Otherwise, you can skip and go straight to the generation phase to respond.
            You do not need to be stringent with the keywords in the question related to these topics.
            Give a binary choice 'web_search' or 'generate' based on the question. 
            Return the JSON with a single key 'choice' with no premable or explanation. 
            
            Question to route: {question} 
            
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
            
            """,
            input_variables=["question"],
        )

        # Chain
        self.question_router = self.router_prompt | self.llama3_json | JsonOutputParser()

        # Query Transformation
        self.query_prompt = PromptTemplate(
            template="""
            
            <|begin_of_text|>
            
            <|start_header_id|>system<|end_header_id|> 
            
            You are an expert at crafting web search queries for research questions.
            More often than not, a user will ask a basic question that they wish to learn more about, however it might not be in the best format. 
            Reword their query to be the most effective web search string possible.
            Return the JSON with a single key 'query' with no premable or explanation. 
            
            Question to transform: {question} 
            
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
            
            """,
            input_variables=["question"],
        )

        # Chain
        self.query_chain = self.query_prompt | self.llama3_json | JsonOutputParser()

        # Graph State
        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                question: question
                generation: LLM generation
                search_query: revised question for web search
                context: web_search result
            """
            question : str
            generation : str
            search_query : str
            context : str

        self.GraphState = GraphState

        # Build the nodes
        self.workflow = StateGraph(self.GraphState)
        self.workflow.add_node("websearch", self.web_search)
        self.workflow.add_node("transform_query", self.transform_query)
        self.workflow.add_node("generate", self.generate)

        # Build the edges
        self.workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "websearch")
        self.workflow.add_edge("websearch", "generate")
        self.workflow.add_edge("generate", END)

        # Compile the workflow
        self.local_agent = self.workflow.compile()

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        
        print("Step: Generating Final Response")
        question = state["question"]
        context = state["context"]

        # Answer Generation
        generation = self.generate_chain.invoke({"context": context, "question": question})
        return {"generation": generation}

    def transform_query(self, state):
        """
        Transform user question to web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended search query
        """
        
        print("Step: Optimizing Query for Web Search")
        question = state['question']
        gen_query = self.query_chain.invoke({"question": question})
        search_query = gen_query["query"]
        return {"search_query": search_query}

    def web_search(self, state):
        """
        Web search based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to context
        """

        search_query = state['search_query']
        print(f'Step: Searching the Web for: "{search_query}"')
        
        # Web search tool call
        search_result = self.web_search_tool.invoke(search_query)
        return {"context": search_result}

    def route_question(self, state):
        """
        route question to web search or generation.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("Step: Routing Query")
        question = state['question']
        output = self.question_router.invoke({"question": question})
        if output['choice'] == "web_search":
            print("Step: Routing Query to Web Search")
            return "websearch"
        elif output['choice'] == 'generate':
            print("Step: Routing Query to Generation")
            return "generate"

    def run_agent(self, query):
        output = self.local_agent.invoke({"question": query})
        print("=======")
        print(output["generation"])
        #print(output)
        #display(Markdown(output["generation"]))


# Test it out!
if __name__ == "__main__":
    chat_module = ChatModule()
    chat_module.run_agent("붕괴 스타레일의 가장 최신 캐릭터는 이름은?")
