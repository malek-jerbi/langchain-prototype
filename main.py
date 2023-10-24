from agents.agent import lookup


def make_request(input: str) -> str:
    agent_answer = lookup(prompt=input)

    return agent_answer


if __name__ == "__main__":
    make_request(input="what is the temperature in Montreal?")

# what is the temperature in montreal
# who is the ceo of optania
# which place is colder, chicoutimi or montreal?
# Give me the weather in Montreal, and tell me who is the CEO of Optania
# does the CEO of optania live in a hot city?
# c'est quoi mon code permanent?