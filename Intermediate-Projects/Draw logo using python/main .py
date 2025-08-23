import turtle as t
import numpy as np
import random
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NeuralPatternGenerator:
    """Simple neural network to generate pattern parameters."""
    
    def __init__(self, input_size: int = 4, hidden_size: int = 8, output_size: int = 3):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def generate_parameters(self, inputs: List[float]) -> Tuple[float, float, float]:
        """Generate angle, distance, and color variation."""
        x = np.array(inputs).reshape(1, -1)
        hidden = self.sigmoid(np.dot(x, self.weights1) + self.bias1)
        output = self.sigmoid(np.dot(hidden, self.weights2) + self.bias2)
        
        # Scale outputs: angle (5-20), distance (10-30), color variation (0-1)
        angle = output[0, 0] * 15 + 5
        distance = output[0, 1] * 20 + 10
        color_var = output[0, 2]
        return angle, distance, color_var

class LogoGenerator:
    """Class to generate AI-enhanced logo patterns using Turtle graphics."""
    
    def __init__(self, turtle: t.Turtle, pattern_generator: NeuralPatternGenerator):
        self.t = turtle
        self.pattern_gen = pattern_generator
        self.colors = ['blue', 'yellow', 'red', 'green', 'purple']
        self.setup_turtle()
    
    def setup_turtle(self) -> None:
        """Initialize turtle settings."""
        self.t.speed(0)
        self.t.hideturtle()
        t.bgcolor('black')
        t.tracer(0)  # Disable animation for faster drawing
    
    def get_dynamic_color(self, base_color: str, variation: float) -> str:
        """Return a dynamic color based on AI variation."""
        if variation > 0.7:
            return random.choice([c for c in self.colors if c != base_color])
        return base_color
    
    def draw_curve(self, angle: float, distance: float, steps: int) -> None:
        """Draw a smooth curve with given angle and distance."""
        for _ in range(steps):
            self.t.right(angle)
            self.t.forward(distance)
    
    def draw_part(self, start_pos: Tuple[int, int], base_color: str, 
                  flip: bool = False) -> None:
        """Draw one part of the logo with AI-generated parameters."""
        try:
            inputs = [random.random() for _ in range(4)]
            angle, distance, color_var = self.pattern_gen.generate_parameters(inputs)
            color = self.get_dynamic_color(base_color, color_var)
            
            self.t.penup()
            self.t.pencolor(color)
            self.t.fillcolor(color)
            self.t.goto(start_pos)
            self.t.pendown()
            self.t.begin_fill()
            
            direction = -1 if flip else 1
            self.t.setheading(180 if flip else 0)
            self.t.forward(100 / 3)
            
            # Draw first curve
            self.draw_curve(angle, distance, 5)
            self.t.forward(distance)
            self.draw_curve(angle / 2, distance, 5)
            self.t.forward(distance * 1.5)
            self.draw_curve(angle, distance, 5)
            
            # Move to next segment
            self.t.goto(0, 100 * direction)
            self.t.goto(0, 110 * direction)
            self.t.goto(100 * direction, 110 * direction)
            self.t.goto(100 * direction, (110 + 100/3) * direction)
            self.t.left(90)
            
            # Draw second curve
            self.draw_curve(angle, distance, 5)
            self.t.forward(distance)
            self.draw_curve(angle / 2, distance, 5)
            self.t.forward(distance * 1.5)
            self.draw_curve(angle, distance, 5)
            self.t.forward(70)
            
            # Draw closing curve
            self.draw_curve(angle, distance, 5)
            self.t.right(5)
            self.t.goto((100 - 30) * direction, 10 * direction)
            self.draw_curve(-angle, distance, 5)
            self.t.left(5)
            
            # Close the shape
            self.t.goto(start_pos)
            self.t.end_fill()
            
            logging.info(f"Drew {color} part at {start_pos}")
            
        except Exception as e:
            logging.error(f"Error drawing part: {str(e)}")
    
    def draw_eyes(self) -> None:
        """Draw white circular eyes."""
        try:
            for pos, direction in [((-70, 130), 1), ((90, -130), -1)]:
                self.t.penup()
                self.t.color('white')
                self.t.goto(pos)
                self.t.pendown()
                self.t.begin_fill()
                self.t.circle(10)
                self.t.end_fill()
            
            logging.info("Eyes drawn successfully")
        
        except Exception as e:
            logging.error(f"Error drawing eyes: {str(e)}")
    
    def generate_logo(self) -> None:
        """Generate the complete logo with AI variations."""
        self.draw_part((-110, -100), 'blue', flip=True)
        self.draw_part((110, 100), 'yellow')
        self.draw_eyes()
        t.update()  # Update the screen
        t.done()

def main():
    """Main function to run the logo generator."""
    try:
        screen = t.Screen()
        turtle = t.Turtle()
        pattern_gen = NeuralPatternGenerator()
        logo = LogoGenerator(turtle, pattern_gen)
        logo.generate_logo()
        screen.exitonclick()
    
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        t.bye()

if __name__ == "__main__":
    main()