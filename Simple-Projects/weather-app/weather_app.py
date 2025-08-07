import requests

def get_weather_data(city, api_key, units="metric"):
    """Fetch weather data for a given city using OpenWeatherMap API."""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city}&units={units}"
    
    try:
        response = requests.get(complete_url)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"Request error: {req_err}"}

def display_weather(data, city, units):
    """Display weather information in a user-friendly format."""
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    
    if data.get("cod") != 200:
        print(f"Error: {data.get('message', 'Unable to fetch weather data')}")
        return
    
    # Extract relevant data
    main = data.get("main", {})
    weather = data.get("weather", [{}])[0]
    wind = data.get("wind", {})
    
    temp = main.get("temp")
    humidity = main.get("humidity")
    description = weather.get("description")
    wind_speed = wind.get("speed")
    
    # Unit symbols
    temp_unit = "°C" if units == "metric" else "°F"
    wind_unit = "m/s" if units == "metric" else "mph"
    
    print(f"\nWeather in {city.title()}:")
    print(f"Temperature: {temp}{temp_unit}")
    print(f"Humidity: {humidity}%")
    print(f"Condition: {description.title()}")
    print(f"Wind Speed: {wind_speed} {wind_unit}")

def main():
    """Main function to run the weather app."""
    api_key = "9adced7ddff5c6dc7f031455d3dec00e"  # Replace with your own API key
    
    while True:
        city = input("\nEnter city name (or 'quit' to exit): ").strip()
        if city.lower() == "quit":
            print("Goodbye!")
            break
            
        if not city:
            print("Error: Please enter a valid city name.")
            continue
            
        # Ask for unit preference
        unit_choice = input("Choose temperature unit (C for Celsius, F for Fahrenheit): ").strip().lower()
        units = "metric" if unit_choice == "c" else "imperial"
        
        # Fetch and display weather
        weather_data = get_weather_data(city, api_key, units)
        display_weather(weather_data, city, units)

if __name__ == "__main__":
    main()