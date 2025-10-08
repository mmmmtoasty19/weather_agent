"""
AI Weather Agent
Inspired by Dave Ebbelaar's single file AI agents
Uses Claude API with tool calling
"""

import os
import json
import time
from typing import Literal, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import requests
from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

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
# Pydancit Models for Structured Data
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
    data: Optional[Any] = Field(default=None, description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")

# ============================================================================
# Weather API Tools
# ============================================================================

def get_current_weather(location: str, units: str="metric") -> ToolResult:
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

        console.print(
            f"[cyan] Fetching current weather for {location}...[/cyan]"
        )
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

