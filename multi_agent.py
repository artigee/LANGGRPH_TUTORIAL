import functools
import operator
from typing import Annotated, Sequence, TypedDict

from colorama import Fore, Style
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from setup_environment import setup_environment_variables
from tools import generate_image, markdown_to_pdf_file

from multi_agent_prompts import (
    TEAM_SUPERVISOR_SYSTEM_PROMPT,
    TRAVEL_AGENT_SYSTEM_PROMPT,
    LANGUAGE_ASSISTANT_SYSTEM_PROMPT,
    VISUALIZER_SYSTEM_PROMPT,
    DESIGNER_SYSTEM_PROMPT,
)

# ------------------------------------------------------------------------

setup_environment_variables("Multi_Agent_Team")

TRAVEL_AGENT_NAME = "travel_agent"
LANGUAGE_ASSISTANT_NAME = "language_assistant"
VISUALIZER_NAME = "visualizer"

DESIGNER_NAME = "designer"
TEAM_SUPERVISOR_NAME = "team_supervisor"

MEMBERS = [TRAVEL_AGENT_NAME, LANGUAGE_ASSISTANT_NAME, VISUALIZER_NAME]
OPTIONS = ["FINISH"] + MEMBERS

TAVILY_TOOL = TavilySearchResults()

LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")

def create_agent(llm: BaseChatModel, tools: list, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor

class AgentState(TypedDict):
    #Annotated is just used as it allows us to add the annotation of operator.add.
    messages: Annotated[Sequence[BaseMessage], operator.add]
    #the name of the next agent to call by the decision from team_supervisor
    next: str

def agent_node(state, agent, name):
    #state object, an agent, and the string name for the agent
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

router_function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "next",
                "anyOf": [
                    {"enum": OPTIONS},
                ],
            }
        },
        "required": ["next"],
    },
}

#we use the join method on the OPTIONS and MEMBERS lists 
# to turn them into a single string with the members separated 
# by a comma and a space as we cannot pass list variables to LLMs.
# system, messages, system ( with options and members )
team_supervisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", TEAM_SUPERVISOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        )
    ]
).partial(options=", ".join(OPTIONS), members=", ".join(MEMBERS))

team_supervisor_chain = (
    team_supervisor_prompt_template
    | LLM.bind_functions(functions=[router_function_def], function_call="route")
    | JsonOutputFunctionsParser(function_call=router_function_def)
)

# Create the agents

travel_agent = create_agent(LLM, [TAVILY_TOOL], TRAVEL_AGENT_SYSTEM_PROMPT)
travel_agent_node = functools.partial(agent_node, agent=travel_agent, name=TRAVEL_AGENT_NAME)

language_assistant = create_agent(LLM, [TAVILY_TOOL], LANGUAGE_ASSISTANT_SYSTEM_PROMPT)
language_assistant_node = functools.partial(agent_node, agent=language_assistant, name=LANGUAGE_ASSISTANT_NAME)

visualizer = create_agent(LLM, [generate_image], VISUALIZER_SYSTEM_PROMPT)
visualizer_node = functools.partial(agent_node, agent=visualizer, name=VISUALIZER_NAME) 

designer = create_agent(LLM, [markdown_to_pdf_file], DESIGNER_SYSTEM_PROMPT)
designer_node = functools.partial(agent_node, agent=designer, name=DESIGNER_NAME)

# Create the workflow

workflow = StateGraph(AgentState)

workflow.add_node(TRAVEL_AGENT_NAME, travel_agent_node)
workflow.add_node(LANGUAGE_ASSISTANT_NAME, language_assistant_node)
workflow.add_node(VISUALIZER_NAME, visualizer_node)
workflow.add_node(DESIGNER_NAME, designer_node)
workflow.add_node(TEAM_SUPERVISOR_NAME, team_supervisor_chain)

for member in MEMBERS:
    workflow.add_edge(member, TEAM_SUPERVISOR_NAME)

workflow.add_edge(DESIGNER_NAME, END)

conditional_map = { name: name for name in MEMBERS }
conditional_map["FINISH"] = DESIGNER_NAME
workflow.add_conditional_edges(
    TEAM_SUPERVISOR_NAME, lambda x: x["next"], conditional_map
)

workflow.set_entry_point(TEAM_SUPERVISOR_NAME)
travel_agent_graph = workflow.compile()

for chunk in travel_agent_graph.stream(
    {"messages": [HumanMessage(content="I want to go to Paris for three days.")]}
):
    if "__end__" not in chunk:
        print(chunk)
        print(f"{Fore.GREEN}#################################{Style.RESET_ALL}")



# add_conditional_edges : Lamda explaination
# ( source, workflow  (workflow : workflow["next"]), destination_map )
# # If team supervisor returns:
# state = {"next": "travel_agent"}
# next_node = lambda x: x["next"]  # Returns "travel_agent"
# destination = conditional_map["travel_agent"]  # Maps to "travel_agent" node

# # Or if supervisor decides to finish:
# state = {"next": "FINISH"}
# next_node = lambda x: x["next"]  # Returns "FINISH"
# destination = conditional_map["FINISH"]  # Maps to "designer" node

# Learning Notes:
# ------------------------------------------------------------------------
# # Common result parameters
# result = agent.invoke(state)
# result["output"]      # The final response/output text
# result["input"]       # The original input to the agent
# result["log"]         # Execution log/trace (if available)
# result["intermediate_steps"]  # Steps taken by agent (if available)

# # Example usage:
# print(result["output"])  # What you're currently using
# print(result.get("log", "No log available"))  # Safe access with default value

# # Agent template usage
# agent_chain = agent_template.invoke({
#     "messages": [
#         HumanMessage(content="What's the weather?"),
#         AIMessage(content="I'll check the weather API..."),
#         # ... more conversation history
#     ],
#     "agent_scratchpad": [
#         AIMessage(content="1. Checking weather API\n2. Processing location...\n")
#     ]
# })

# template = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful AI bot. Your name is {name}."),
#             ("human", "Hello, how are you doing?"),
#             ("ai", "I'm doing well, thanks!"),
#             ("human", "{user_input}"),
#         ])

# template = ChatPromptTemplate.from_messages([
#     ("human", "Please tell me the french and german words for {word} with an example sentence for each.")
# ])

# The convention in LangChain's documentation and examples often uses:
# "messages when working with agents
# "chat_history when working with memory components
# But they serve the same purpose: storing the conversation history.
# Just make sure the variable name matches what you use in your code!
# Example:
# If using "messages":
# chain.invoke({
#     "messages": [
#         HumanMessage(content="Hello"),
#         AIMessage(content="Hi there!")
#     ]
# })
# # If using "chat_history":
# chain.invoke({
#     "chat_history": [
#         HumanMessage(content="Hello"),
#         AIMessage(content="Hi there!")
#     ]
# })


# Benefit of BaseChatModel:
# # Any of these will work because they inherit from BaseChatModel
# llm1 = ChatOpenAI()
# llm2 = ChatAnthropic()
# llm3 = ChatBard()
# create_agent(llm1, ...)  # Works
# create_agent(llm2, ...)  # Works
# create_agent(llm3, ...)  # Works


# AgentExecutor vs ToolExecutor difference
#
# AgentExecutor example
# agent_executor = AgentExecutor(
#     agent=agent,  # The agent that makes decisions
#     tools=tools   # Available tools for the agent to use
# )
# # Usage
# agent_executor.invoke({
#     "input": "What's the weather and should I bring an umbrella?",
#     "messages": [...],  # conversation history
# })
#
# # Agent can: 
# # 1. Check weather tool
# # 2. Analyze results
# # 3. Make a recommendation about umbrella
#
# # ToolExecutor example
# tool_executor = ToolExecutor(tool=weather_tool)
#
# # Usage
# tool_executor.invoke({
#     "input": "What's the weather?",
# })
# # Can only: Execute the weather tool directly
