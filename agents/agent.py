from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from tools.tools import google_search
from sql_agent import sql_agent


def lookup(prompt: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    tools = load_tools(["openweathermap-api"], llm)
    tools.extend([
        Tool(
            name="Crawl Google",
            func=google_search,
            description="useful for when you need to search on google",
        ),
        Tool(
            name="personal info agent",
            func=sql_agent,
            description="useful for when you need to retrieve any personal data. this is another agent, so the input can be in natural language.",
        )
    ]
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_answer = agent.run(prompt)

    return prompt_answer
