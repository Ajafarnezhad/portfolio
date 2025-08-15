# Weather App: Stay Updated with Real-Time Weather 🌤️✨

Welcome to the **Weather App**, a simple yet powerful Python program that fetches and displays current weather information for any city using the OpenWeatherMap API. With support for Celsius or Fahrenheit, detailed weather metrics, and a user-friendly interface, this project is a perfect portfolio piece to showcase your skills in API integration, error handling, and interactive application design.

---

## 🌟 Project Highlights
This project delivers real-time weather data with a clean and intuitive interface, allowing users to check temperature, humidity, wind speed, and more. Featuring robust error handling and multiple query support, it’s ideal for demonstrating Python programming and API-driven application development.

---

## 🚀 Features
- **Real-Time Weather Data**: Fetches current weather information for any city via the OpenWeatherMap API.
- **Temperature Units**: Supports user-selected temperature units (Celsius or Fahrenheit).
- **Detailed Output**: Displays temperature, humidity, wind speed, and weather description (e.g., "clear sky").
- **Error Handling**: Robustly handles invalid city names, API issues, and connectivity errors with clear feedback.
- **Multiple Queries**: Allows users to check weather for multiple cities until they choose to exit.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `requests`

Install dependencies with:
```bash
pip install requests
```

---

## 🎮 How to Install
1. Ensure Python 3.8+ is installed.
2. Install the required library:
   ```bash
   pip install requests
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   ```
4. Navigate to the project directory:
   ```bash
   cd portfolio/Simple-Projects/weather-app
   ```
5. Get a free API key from [OpenWeatherMap](https://openweathermap.org/api).
6. Replace `api_key` in `weather_app.py` with your own API key.

---

## 🎯 How to Run
1. Run the script:
   ```bash
   python weather_app.py
   ```
2. Follow the prompts to:
   - Enter a city name (e.g., "Tehran").
   - Choose a temperature unit (C for Celsius, F for Fahrenheit).
3. View the weather details and choose to query another city or exit.

---

## 📈 Example Interaction
```
Welcome to Weather App!
Enter city name (or 'quit' to exit): Tehran
Choose temperature unit (C for Celsius, F for Fahrenheit): C
Weather in Tehran:
Temperature: 25.3°C
Humidity: 60%
Wind Speed: 3.5 m/s
Description: Clear sky
Would you like to check another city? (y/n): n
```

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **GUI Interface**: Add a Tkinter or PyQt interface for a graphical experience.
- **Weather Forecasts**: Extend to include 5-day weather forecasts or hourly updates.
- **Visualization**: Generate charts for temperature or humidity trends using `matplotlib` or `plotly`.
- **Location Detection**: Auto-detect user location using geolocation APIs.
- **Unit Testing**: Implement `pytest` for robust validation of API calls and input handling.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Stay informed about the weather with the **Weather App** and showcase your API integration skills! 🚀