Age Calculator
This is a simple Python program that calculates your age based on your birth date and converts it into various units: years, months, days, hours, and seconds. The calculation is precise, considering leap years and exact days.
Features

Accurate age calculation using birth date.
Handles leap years correctly.
Converts age to months, days, hours, and seconds.
Error handling for invalid date inputs.

Installation

Ensure you have Python 3.x installed.
No external libraries are required (uses standard datetime module).
Clone the repository:git clone https://github.com/Ajafarnezhad/portfolio.git


Navigate to the project directory:cd portfolio/Simple-Projects/age-calculator



Usage
Run the script and provide your birth date when prompted:
python age_calculator.py

Example input:

Birth year: 1990
Birth month: 5
Birth day: 15

Sample output (based on August 7, 2025):
Your age is 35 years, or 422 months, or 12877 days, or 309048 hours, or 1112572800 seconds.

How It Works

The program uses the datetime module to handle dates.
It calculates the exact number of days between birth and current date.
Months are approximated based on years and calendar months.
Hours and seconds are derived from total days.

Improvements Ideas

Add GUI using Tkinter for a user-friendly interface.
Support for time of birth for even more precision (e.g., hours and minutes).
Web version using Flask or Django.

This project is part of my portfolio. Check out my other projects on GitHub: Ajafarnezhad
License: MIT (Free to use and modify)