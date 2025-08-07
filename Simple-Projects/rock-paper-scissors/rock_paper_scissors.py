import random

def get_user_choice():
    """Get and validate user's choice."""
    choices = ["rock", "paper", "scissors"]
    while True:
        choice = input("Enter your choice (rock, paper, scissors, or 'quit' to exit): ").lower().strip()
        if choice == "quit":
            return None
        if choice in choices:
            return choice
        print("Invalid choice! Please choose rock, paper, scissors, or quit.")

def get_computer_choice():
    """Generate computer's choice."""
    return random.choice(["rock", "paper", "scissors"])

def determine_winner(user_choice, computer_choice):
    """Determine the winner of a round based on game rules."""
    if user_choice == computer_choice:
        return "tie", None
    
    winning_combinations = {
        ("rock", "scissors"): "Rock crushes scissors!",
        ("paper", "rock"): "Paper covers rock!",
        ("scissors", "paper"): "Scissors cut paper!"
    }
    
    if (user_choice, computer_choice) in winning_combinations:
        return "user", winning_combinations[(user_choice, computer_choice)]
    else:
        # Reverse the combination to get computer's winning message
        reverse_message = winning_combinations.get((computer_choice, user_choice), "")
        return "computer", reverse_message

def play_game(max_score=3):
    """Main function to play the Rock, Paper, Scissors game."""
    user_score = 0
    computer_score = 0
    
    print(f"\nRock, Paper, Scissors! First to {max_score} points wins.")
    
    while user_score < max_score and computer_score < max_score:
        print(f"\nScore - You: {user_score}, Computer: {computer_score}")
        
        user_choice = get_user_choice()
        if user_choice is None:
            print("Game ended early.")
            break
            
        computer_choice = get_computer_choice()
        print(f"You chose: {user_choice.title()} | Computer chose: {computer_choice.title()}")
        
        result, message = determine_winner(user_choice, computer_choice)
        if result == "tie":
            print("It's a tie!")
        elif result == "user":
            user_score += 1
            print(f"You win this round! {message}")
        else:
            computer_score += 1
            print(f"Computer wins this round! {message}")
    
    # Display final result
    print(f"\nFinal Score - You: {user_score}, Computer: {computer_score}")
    if user_score > computer_score:
        print("🎉 YOU WIN! 🎉")
    elif computer_score > user_score:
        print("😔 YOU LOSE 😔")
    else:
        print("🤝 It's a tie! 🤝")

def main():
    """Main entry point to start or replay the game."""
    while True:
        try:
            max_score = int(input("Enter max score to win (e.g., 3): "))
            if max_score <= 0:
                print("Please enter a positive number.")
                continue
            play_game(max_score)
        except ValueError:
            print("Please enter a valid number.")
            continue
            
        replay = input("\nPlay again? (yes/no): ").lower().strip()
        if replay != "yes":
            print("Thanks for playing! Goodbye!")
            break

if __name__ == "__main__":
    main()