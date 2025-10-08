"""
AI Weather Agent
Inspired by Dave Ebbelaar's single file AI agents
Uses Claude API with tool calling
"""

import json
import os
from datetime import datetime
from typing import Any

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Import enviromental variables
load_dotenv()

# Initialize clients
console = Console()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5"
MODEL_NAME = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 10

# ============================================================================
# Pydantic Models for Structured Data
# ============================================================================


class WeatherData(BaseModel):
    """Model for weather data response"""

    location: str = Field(description="City name")
    country: str = Field(description="Country code")
    temperature: float = Field(description="Current temperature")
    feels_like: float = Field(description="Feels like temperature")
    temp_min: float = Field(description="Minimum temperature")
    temp_max: float = Field(description="Maximum temperature")
    humidity: int = Field(description="Humidity percentage")
    pressure: int = Field(description="Atmospheric pressure")
    description: str = Field(description="Weather description")
    main: str = Field(description="Main weather condition")
    wind_speed: float = Field(description="Wind speed")
    clouds: int = Field(description="Cloudiness percentage")
    visibility: int = Field(description="Visibility in meters")
    sunrise: str = Field(description="Sunrise time")
    sunset: str = Field(description="Sunset time")
    units: str = Field(description="Temperature units (metric/imperial)")
    timestamp: str = Field(description="Data timestamp")


class ForecastItem(BaseModel):
    """Single forecast item"""

    datetime: str = Field(description="Forecast datetime")
    temperature: float = Field(description="Temperature")
    description: str = Field(description="Weather description")
    wind_speed: float = Field(description="Wind speed")
    humidity: int = Field(description="Humidity percentage")


class ForecastData(BaseModel):
    """Structured forecast data response"""

    location: str = Field(description="City name")
    country: str = Field(description="Country code")
    forecasts: list[ForecastItem] = Field(description="List of forecast items")


class ToolResult(BaseModel):
    """Result from tool execution"""

    success: bool = Field(description="Whether tool execution succeeded")
    data: Any | None = Field(default=None, description="Tool output data")
    error: str | None = Field(default=None, description="Error message if failed")


# ============================================================================
# Weather API Tools
# ============================================================================


def get_current_weather(location: str, units: str = "metric") -> ToolResult:
    """
    Fetch current weather data for a location using OpenWeatherMap API.

    Args:
        location (str): City name.
        units (str): Units for temperature ('metric' or 'imperial').

    Returns:
        ToolResult: Result containing weather data or error message.
    """
    try:
        location = location.strip()

        console.print(f"[cyan] Fetching current weather for {location}...[/cyan]")
        url = f"{WEATHER_API_URL}/weather"
        params = {"q": location, "appid": WEATHER_API_KEY, "units": units}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        weather_data = WeatherData(
            location=data["name"],
            country=data["sys"]["country"],
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            temp_min=data["main"]["temp_min"],
            temp_max=data["main"]["temp_max"],
            humidity=data["main"]["humidity"],
            pressure=data["main"]["pressure"],
            description=data["weather"][0]["description"],
            main=data["weather"][0]["main"],
            wind_speed=data["wind"]["speed"],
            clouds=data["clouds"]["all"],
            visibility=data.get("visibility", 0),
            sunrise=datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%H:%M"),
            sunset=datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%H:%M"),
            units=units,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        console.print("[green] Weather data retrieved successfully[/green]")

        return ToolResult(success=True, data=weather_data.model_dump())

    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch weather data: {str(e)}"
        console.print(f"[red] {error_msg}[/red]")
        return ToolResult(success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        console.print(f"[red] {error_msg}[/red]")
        return ToolResult(success=False, error=error_msg)


def get_weather_forecast(location: str, units: str = "metric") -> ToolResult:
    """
    Fetch 5-day weather forecast a location using OpenWeatherMap API.

    Args:
        location (str): City name.
        units (str): Units for temperature ('metric' or 'imperial').

    Returns:
        ToolResult: Result containing weather data or error message.
    """
    try:
        location = location.strip()

        console.print(f"[cyan] Fetching forecast for {location}...[/cyan]")
        url = f"{WEATHER_API_URL}/forecast"
        params = {"q": location, "appid": WEATHER_API_KEY, "units": units}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        forecasts = []
        for item in data["list"][:40]:
            forecasts.append(
                ForecastItem(
                    datetime=item["dt_txt"],
                    temperature=item["main"]["temp"],
                    description=item["weather"][0]["description"],
                    wind_speed=item["wind"]["speed"],
                    humidity=item["main"]["humidity"],
                )
            )

        forecast_data = ForecastData(
            location=data["city"]["name"],
            country=data["city"]["country"],
            forecasts=forecasts,
        )

        console.print("[green] Forecast data retrieved successfully[/green]")

        return ToolResult(success=True, data=forecast_data.model_dump())

    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch weather data: {str(e)}"
        console.print(f"[red] {error_msg}[/red]")
        return ToolResult(success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        console.print(f"[red] {error_msg}[/red]")
        return ToolResult(success=False, error=error_msg)


# ============================================================================
# Tool Definitions for Claude
# ============================================================================

TOOLS = [
    {
        "name": "get_current_weather",
        "description": (
            "Fetches current weather data for a specified location. "
            "Use this when the user asks about current weather conditions, "
            "temperature, humidity, wind, or any present-moment weather information. "
            "Supports precise locations using 'city,state,country' "
            "format for small towns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": 'Location in one of these formats: "City", "City,Country", or "City,State,Country". Examples: "London", "Paris,FR", "Austin,TX,US", "Bladenboro,NC,US". For small towns, use the full "City,State,Country" format for best results. Use ISO 3166 country codes (US, GB, FR, etc.) and standard 2-letter state codes (TX, CA, NC, etc.)',
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": 'Temperature units: "metric" for Celsius, "imperial" for Fahrenheit. Default is metric.',
                    "default": "metric",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_weather_forecast",
        "description": "Fetches weather forecast data (next 5 days in 3-hour intervals) for a specified location. Use this when the user asks about future weather, forecasts, upcoming conditions, or what the weather will be like. Supports precise locations using 'city,state,country' format for small towns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": 'Location in one of these formats: "City", "City,Country", or "City,State,Country". Examples: "London", "Paris,FR", "Austin,TX,US", "Bladenboro,NC,US". For small towns, use the full "City,State,Country" format for best results. Use ISO 3166 country codes (US, GB, FR, etc.) and standard 2-letter state codes (TX, CA, NC, etc.)',
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": 'Temperature units: "metric" for Celsius, "imperial" for Fahrenheit. Default is metric.',
                    "default": "metric",
                },
            },
            "required": ["location"],
        },
    },
]

TOOL_MAP = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
}

# ============================================================================
# Agent Loop
# ============================================================================


def process_tool_call(tool_name: str, tool_input: dict) -> ToolResult:
    """Execute a tool and return the result"""
    if tool_name not in TOOL_MAP:
        return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

    tool_func = TOOL_MAP[tool_name]
    return tool_func(**tool_input)


def run_agent(user_message: str) -> str:
    """
    Run the agentic loop: send message to claude, process tool calls, return response.

    Args:
        user_message: Users input query

    Returns:
        Claudes's final response as a string
    """
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(MAX_ITERATIONS):
        console.print(f"\n[dim]--- Iteration {iteration + 1} ---[/dim]")

        # Call Claude API
        response = client.messages.create(
            model=MODEL_NAME, max_tokens=4096, tools=TOOLS, messages=messages
        )

        # TODO This is for Debugging, Can remove later
        console.print(f"[dim]Stop reason: {response.stop_reason}[/dim]")

        # Check is Claude needs a tool
        if response.stop_reason == "tool_use":
            assistant_message = {"role": "assistant", "content": response.content}
            messages.append(assistant_message)

            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    console.print(
                        f"[yellow] Calling tool: {tool_name}({json.dumps(tool_input, indent=2)})[/yellow]"
                    )

                    result = process_tool_call(tool_name, tool_input)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                {
                                    "success": result.success,
                                    "data": result.data,
                                    "error": result.error,
                                }
                            ),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

            # Continue the loop to get Claudes next response
        elif response.stop_reason == "end_turn":
            # Claude finished the loop and provided a response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text

            return final_response

        else:
            return f"Unexpected stop reason: {response.stop_reason}"

    return "Maximum iterations reached. Please try again with a more specific query"


# ============================================================================
# Main Application
# ============================================================================


def print_welcome():
    """Display welcome message"""
    welcome_text = """
    # AI Weather Agent

    Hello!  I am an intelligent weather assistant powered by Claude.

    **What I can do:**
    - Get current weather conditions for any city
    - Provide weather forecasts (5 days Max)
    - Answer questions about the weather
    - Support for both Celsuis and Fahrenheit

    **Example Queries:**
    - "What's the weather like in Paris?"
    - "Will it rain in Tokyo tomorrow?"
    - "What should I pack for my trip to Londan in three days?"

    **Commands**
    - Type 'quit', 'exit' or 'bye' to leave
    - Use natural language, I can figure it out!

    *Powered by Claude and OpenWeatherMap*
    """
    console.print(Panel(Markdown(welcome_text), border_style="blue"))


def main():
    # TODO add API Checks

    print_welcome()

    while True:
        try:
            console.print("\n[bold cyan]You:[/bold cyan]", end=" ")
            user_input = input().strip()

            if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                console.print(
                    "\n[bold green]Goodbye! Enjoy the weather![/bold green]\n"
                )
                break

            # TODO Should I remind the user to type?
            if not user_input:
                continue

            console.print("\n[bold yellow]Assistant:[/bold yellow]")
            response = run_agent(user_input)
            console.print(f"[white]{response}[/white]")

        except KeyboardInterrupt:
            console.print("\n\n[bold red] Interrupted. Goodbye[/bold red]\n")
            break

        except Exception as e:
            console.print(f"\n[bold red] Error: {e}[/bold red]")
            continue


if __name__ == "__main__":
    main()
