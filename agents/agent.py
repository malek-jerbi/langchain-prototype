from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tools.tools import tools


def lookup(prompt: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )
    prompt_answer = agent.run(prompt)

    return prompt_answer
