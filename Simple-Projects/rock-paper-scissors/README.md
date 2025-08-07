Rock, Paper, Scissors

This is a simple Python implementation of the classic Rock, Paper, Scissors game. Play against the computer, choose your moves, and try to reach the target score first!

Features



Play Rock, Paper, Scissors with a customizable winning score.

Validates user input to prevent errors.

Displays clear round results with descriptive messages (e.g., "Rock crushes scissors!").

Supports replaying the game without restarting the script.

Tracks and displays scores for both player and computer.



Installation



Ensure you have Python 3.x installed.

No external libraries are required (uses standard random module).

Clone the repository:git clone https://github.com/Ajafarnezhad/portfolio.git





Navigate to the project directory:cd portfolio/Simple-Projects/rock-paper-scissors







Usage

Run the script and follow the prompts:

python rock\_paper\_scissors.py



Example interaction:

Enter max score to win (e.g., 3): 3

Rock, Paper, Scissors! First to 3 points wins.



Score - You: 0, Computer: 0

Enter your choice (rock, paper, scissors, or 'quit' to exit): rock

You chose: Rock | Computer chose: Scissors

You win this round! Rock crushes scissors!



Score - You: 1, Computer: 0

...

Final Score - You: 3, Computer: 1

🎉 YOU WIN! 🎉

Play again? (yes/no): no

Thanks for playing! Goodbye!



How It Works



Uses the random module to generate the computer's choice.

Validates user input to ensure only valid choices (rock, paper, scissors, quit) are accepted.

Determines the winner using a dictionary of winning combinations.

Tracks scores and ends the game when a player reaches the max score.

Supports replaying with a new max score.



Improvements Ideas



Add a GUI using Tkinter or Pygame for a visual interface.

Include sound effects for wins/losses using a library like playsound.

Add a multiplayer mode to play against another human.



This project is part of my portfolio. Check out my other projects on GitHub: Ajafarnezhad

License: MIT (Free to use and modify)

