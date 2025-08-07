import pyautogui
import time
import random
import sys

# Configure pyautogui safety features
pyautogui.FAILSAFE = True  # Move mouse to top-left corner to stop the script
pyautogui.PAUSE = 0.1  # Small pause between actions for safety

def get_random_love_message():
    """Return a random love message from a predefined list."""
    messages = [
        "I love you to the moon and back!",
        "You're my sunshine on a cloudy day!",
        "My heart beats for you!",
        "You're the best thing that's ever happened to me!",
        "I adore you more than words can say!"
    ]
    return random.choice(messages)

def send_messages(message, count, delay, random_mode=False):
    """Send the specified message repeatedly with a given delay."""
    print(f"Starting in 3 seconds... Move mouse to top-left corner to stop.")
    time.sleep(3)  # Give user time to focus on the target application
    
    try:
        for i in range(count):
            current_message = get_random_love_message() if random_mode else message
            pyautogui.write(current_message, interval=0.05)
            pyautogui.press("enter")
            print(f"Sent message {i+1}/{count}: {current_message}")
            time.sleep(delay)
    except pyautogui.FailSafeException:
        print("Script stopped by moving mouse to top-left corner.")
    except KeyboardInterrupt:
        print("Script stopped by user (Ctrl+C).")

def main():
    """Main function to run the love bot."""
    print("Welcome to Love Bot! This script types and sends love messages.")
    
    try:
        # Get user inputs
        mode = input("Choose mode (1 for custom message, 2 for random messages): ").strip()
        if mode not in ["1", "2"]:
            print("Invalid mode! Please choose 1 or 2.")
            return
            
        random_mode = (mode == "2")
        message = input("Enter your message: ").strip() if not random_mode else ""
        if not random_mode and not message:
            print("Error: Message cannot be empty in custom mode.")
            return
            
        count = int(input("Enter number of messages to send: "))
        if count <= 0:
            print("Error: Number of messages must be positive.")
            return
            
        delay = float(input("Enter delay between messages (seconds, e.g., 0.5): "))
        if delay < 0:
            print("Error: Delay cannot be negative.")
            return
            
        # Confirm before starting
        print(f"\nSending {count} messages with {delay}s delay...")
        send_messages(message, count, delay, random_mode)
        
        print("\nDone! All messages sent.")
        
    except ValueError:
        print("Error: Please enter valid numbers for count and delay.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")