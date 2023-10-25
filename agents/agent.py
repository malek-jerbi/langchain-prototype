from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from tools.tools import google_search
from sql_agent import sql_agent


def lookup(prompt: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    tools = load_tools(["openweathermap-api"], llm)
    tools.extend([
        Tool(
            name="Crawl_Google",
            func=google_search,
            description="useful for when you need to search on google",
        ),
        Tool(
            name="personal_info_agent",
            func=sql_agent,
            description="useful for when you need to administrative personal data, for example, social insurance number. this is another agent, so the input can be in natural language.",
        )
    ]
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )
    prompt_answer = agent.run(prompt)

    return prompt_answer
