import numpy as np
import tkinter as tk
import random

class NeuralNetwork:
    def __init__(self):
        self.input_weights = np.random.rand(4, 4) - 0.5
        self.output_weights = np.random.rand(4, 1) - 0.5

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden_layer = self.sigmoid(np.dot(inputs, self.input_weights))
        output_layer = self.sigmoid(np.dot(hidden_layer, self.output_weights))
        return output_layer > 0.5

    def copy_and_mutate(self, mutation_rate=0.1):
        new_nn = NeuralNetwork()
        new_nn.input_weights = self.input_weights + np.random.randn(4, 4) * mutation_rate
        new_nn.output_weights = self.output_weights + np.random.randn(4, 1) * mutation_rate
        return new_nn

# Constants
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
BIRD_RADIUS = 10
BIRD_X = 50
GRAVITY = 2
JUMP_STRENGTH = -20
OBSTACLE_WIDTH = 50
OBSTACLE_GAP = 150
OBSTACLE_SPEED = 5
NUM_BIRDS = 100
TOP_BIRDS = 4
MUTATION_RATE = 0.05

class Bird:
    def __init__(self, canvas, brain):
        self.canvas = canvas
        self.y = WINDOW_HEIGHT // 2
        self.velocity = 0
        self.alive = True
        self.brain = brain
        self.distance_flown = 0
        self.shape = self.canvas.create_oval(
            BIRD_X - BIRD_RADIUS, self.y - BIRD_RADIUS,
            BIRD_X + BIRD_RADIUS, self.y + BIRD_RADIUS,
            fill="yellow"
        )

    def jump(self):
        self.velocity = JUMP_STRENGTH

    def apply_gravity(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.distance_flown += OBSTACLE_SPEED
        self.canvas.coords(self.shape, BIRD_X - BIRD_RADIUS, self.y - BIRD_RADIUS, BIRD_X + BIRD_RADIUS, self.y + BIRD_RADIUS)

    def reset(self):
        self.y = WINDOW_HEIGHT // 2
        self.velocity = 0
        self.alive = True
        self.distance_flown = 0
        self.canvas.coords(self.shape, BIRD_X - BIRD_RADIUS, self.y - BIRD_RADIUS, BIRD_X + BIRD_RADIUS, self.y + BIRD_RADIUS)

    def decide(self, inputs):
        if self.brain.forward(inputs) and self.alive:
            self.jump()

class FlappyBirdGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Flappy Bird Game")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="sky blue")
        self.canvas.pack()

        # Game state
        self.birds = [Bird(self.canvas, NeuralNetwork()) for _ in range(NUM_BIRDS)]
        self.obstacles = []
        self.score = 0
        self.iteration = 1
        self.game_over = False

        # Initialize obstacles
        self.create_obstacle()

        # Score and iteration text
        self.score_text = self.canvas.create_text(10, 10, anchor="nw", font=("Arial", 16), text=f"Score: {self.score}")
        self.iteration_text = self.canvas.create_text(10, 30, anchor="nw", font=("Arial", 16), text=f"Iteration: {self.iteration}")

        self.update_game()

    def start_game(self):
        for bird in self.birds:
            if bird.alive:
                bird.jump()

    def reset_game(self):
        # Clear all obstacles from the canvas and reset the obstacles list
        for top_obstacle, bottom_obstacle in self.obstacles:
            self.canvas.delete(top_obstacle)
            self.canvas.delete(bottom_obstacle)
        self.obstacles.clear()
        
        # Clear all birds from the canvas
        for bird in self.birds:
            self.canvas.delete(bird.shape)
            
        # Evolve the birds by creating a new population
        self.birds = self.evolve_birds()
        
        # Reset score and iteration display
        self.score = 0
        self.game_over = False
        self.canvas.itemconfig(self.score_text, text=f"Score: {self.score}")
        self.canvas.itemconfig(self.iteration_text, text=f"Iteration: {self.iteration}")
        
        # Reinitialize obstacles
        self.create_obstacle()

    def evolve_birds(self):
        # Sort birds by distance flown, take the top performers
        top_birds = sorted(self.birds, key=lambda b: b.distance_flown, reverse=True)[:TOP_BIRDS]
        
        # Create new population with mutations
        new_birds = []
        for bird in top_birds:
            for _ in range(NUM_BIRDS // TOP_BIRDS):
                mutated_brain = bird.brain.copy_and_mutate(MUTATION_RATE)
                new_birds.append(Bird(self.canvas, mutated_brain))
        
        self.iteration += 1
        return new_birds

    def create_obstacle(self):
        gap_y = random.randint(OBSTACLE_GAP, WINDOW_HEIGHT - OBSTACLE_GAP)
        top_obstacle = self.canvas.create_rectangle(
            WINDOW_WIDTH, 0, WINDOW_WIDTH + OBSTACLE_WIDTH, gap_y - OBSTACLE_GAP // 2, fill="green"
        )
        bottom_obstacle = self.canvas.create_rectangle(
            WINDOW_WIDTH, gap_y + OBSTACLE_GAP // 2, WINDOW_WIDTH + OBSTACLE_WIDTH, WINDOW_HEIGHT, fill="green"
        )
        self.obstacles.append((top_obstacle, bottom_obstacle))

    def move_obstacles(self):
        for top_obstacle, bottom_obstacle in self.obstacles:
            self.canvas.move(top_obstacle, -OBSTACLE_SPEED, 0)
            self.canvas.move(bottom_obstacle, -OBSTACLE_SPEED, 0)

        if self.canvas.coords(self.obstacles[0][0])[2] < 0:
            self.canvas.delete(self.obstacles[0][0])
            self.canvas.delete(self.obstacles[0][1])
            self.obstacles.pop(0)
            self.create_obstacle()
            self.score += 1
            self.canvas.itemconfig(self.score_text, text=f"Score: {self.score}")

    def check_collision(self):
        alive_birds = 0
        for bird in self.birds:
            if bird.alive:
                alive_birds += 1
                if bird.y + BIRD_RADIUS >= WINDOW_HEIGHT or bird.y - BIRD_RADIUS <= 0:
                    bird.alive = False

                bird_coords = self.canvas.bbox(bird.shape)
                for top_obstacle, bottom_obstacle in self.obstacles:
                    if self.canvas.bbox(top_obstacle) and self.canvas.bbox(bottom_obstacle):
                        if (self.canvas.bbox(top_obstacle)[2] >= bird_coords[0] and 
                            self.canvas.bbox(top_obstacle)[0] <= bird_coords[2]):
                            if (self.canvas.bbox(top_obstacle)[3] >= bird_coords[1] or
                                self.canvas.bbox(bottom_obstacle)[1] <= bird_coords[3]):
                                bird.alive = False
                                self.canvas.delete(bird.shape) 
                                break

        if alive_birds == 0:
            self.game_over = True

    def update_game(self):
        if not self.game_over:
            obstacle = self.obstacles[0]
            obstacle_coords = self.canvas.coords(obstacle[0])

            for bird in self.birds:
                if bird.alive:
                    inputs = np.array([bird.y, obstacle_coords[3], self.canvas.coords(obstacle[1])[1], obstacle_coords[0]])
                    bird.decide(inputs)
                    bird.apply_gravity()

            self.move_obstacles()
            self.check_collision()
        else:
            self.reset_game()

        self.root.after(2, self.update_game)

# Run the game
root = tk.Tk()
game = FlappyBirdGame(root)
root.mainloop()
