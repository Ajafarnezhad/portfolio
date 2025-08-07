from datetime import datetime

# Function to check if a year is a leap year
def is_leap_year(year):
    """Determine if the given year is a leap year."""
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    return False

# Function to get the number of days in a month
def days_in_month(month, year):
    """Return the number of days in the given month and year."""
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    else:
        raise ValueError("Invalid month")

# Function to calculate age in various units
def calculate_age(birth_date, current_date):
    """Calculate age from birth date to current date in years, months, days, hours, and seconds."""
    # Calculate years
    years = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        years -= 1

    # Calculate total days
    total_days = (current_date - birth_date).days

    # Calculate months (approximate, as months vary in length)
    months = years * 12 + (current_date.month - birth_date.month)
    if current_date.day < birth_date.day:
        months -= 1

    # Calculate hours and seconds
    hours = total_days * 24
    seconds = total_days * 24 * 3600  # Ignoring leap seconds for simplicity

    return years, months, total_days, hours, seconds

# Main function to run the program
def main():
    """Main entry point to get user input and display age calculations."""
    try:
        # Get birth date input
        birth_year = int(input("Enter your birth year (YYYY): "))
        birth_month = int(input("Enter your birth month (1-12): "))
        birth_day = int(input("Enter your birth day (1-31): "))

        # Validate birth date
        birth_date = datetime(birth_year, birth_month, birth_day)

        # Get current date
        current_date = datetime.now()

        # Calculate age
        years, months, days, hours, seconds = calculate_age(birth_date, current_date)

        # Output results
        print(f"Your age is {years} years, or {months} months, or {days} days, "
              f"or {hours} hours, or {seconds} seconds.")

    except ValueError as e:
        print(f"Error: {e}. Please enter valid date inputs.")

# Run the program if executed directly
if __name__ == "__main__":
    main()