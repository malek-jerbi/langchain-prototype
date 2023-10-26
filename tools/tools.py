from langchain.serpapi import SerpAPIWrapper
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from langchain.utilities.openweathermap import OpenWeatherMapAPIWrapper
from agents.agent import sql_agent


class WeatherInput(BaseModel):
    location: str = Field(
        description="Input should be a location string (e.g. London,GB).")


@tool("get_weather_by_location", args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """A wrapper around OpenWeatherMap API. Useful for fetching current weather information for a specified location. """
    return OpenWeatherMapAPIWrapper().run(location)


class GoogleSearchInput(BaseModel):
    input: str = Field(
        description="Text based input for google search.")


@tool("google_search", args_schema=GoogleSearchInput)
def google_search(input: str) -> str:
    """A wrapper around SerpAPO. Useful for making google searches. """
    return SerpAPIWrapper().run(input)


class SQL_agentInput(BaseModel):
    input: str = Field(
        description="Natural language text based input for SQL agent.")


@tool("prompt_sql_agent", args_schema=SQL_agentInput)
def prompt_sql_agent(input: str) -> str:
    """Agent that returns personal data. useful for when you need administrative data, for example, social insurance number."""
    return sql_agent(input)


tools = [get_weather, google_search, prompt_sql_agent]
