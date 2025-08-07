Weather App

This is a simple Python program that fetches and displays current weather information for any city using the OpenWeatherMap API. It supports temperature in Celsius or Fahrenheit and provides details like humidity, wind speed, and weather conditions.

Features



Fetches real-time weather data for any city.

Supports temperature units in Celsius or Fahrenheit (user choice).

Displays temperature, humidity, wind speed, and weather description.

Robust error handling for invalid city names or API issues.

Allows multiple queries until the user chooses to exit.



Installation



Ensure you have Python 3.x installed.

Install the required library:pip install requests





Clone the repository:git clone https://github.com/Ajafarnezhad/portfolio.git





Navigate to the project directory:cd portfolio/Simple-Projects/weather-app







Usage



Get a free API key from OpenWeatherMap (the default key may not work).

Replace api\_key in weather\_app.py with your own key.

Run the script:python weather\_app.py





Enter a city name and choose temperature unit (C or F).



Example interaction:

Enter city name (or 'quit' to exit): London

Choose temperature unit (C for Celsius, F for Fahrenheit): C



Weather in London:

Temperature: 15.2°C

Humidity: 77%

Condition: Broken Clouds

Wind Speed: 4.1 m/s



How It Works



Uses the requests library to query the OpenWeatherMap API.

Parses JSON response to extract weather details.

Handles errors for invalid cities, network issues, or API limits.

Supports both metric (Celsius, m/s) and imperial (Fahrenheit, mph) units.



Improvements Ideas



Add a GUI using Tkinter or PyQt for a user-friendly interface.

Include weather forecasts (e.g., 5-day forecast using another OpenWeatherMap endpoint).

Cache API responses to reduce calls and handle rate limits.



This project is part of my portfolio. Check out my other projects on GitHub: Ajafarnezhad

License: MIT (Free to use and modify)

