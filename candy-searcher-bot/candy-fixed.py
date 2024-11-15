import math
import numpy as np

BOARD_SIZE = 16
NUM_BOTS = 100
NUM_CANDIES = 1
ROUNDS_PER_GAME = 24
GAME_ITERATIONS = 300
MUTATION_RATE = 0.05
STARTING_COORDS_BOT = [BOARD_SIZE // 4, BOARD_SIZE // 4]
STARTING_COORDS_CANDY = [BOARD_SIZE // 4 * 3, BOARD_SIZE // 4 * 3]

class NeuralNetwork:
    def __init__(self, input_size=BOARD_SIZE**2, hidden_size=BOARD_SIZE*2, output_size=4):
        self.input_weights = np.random.rand(input_size, hidden_size) - 0.5
        self.output_weights = np.random.rand(hidden_size, output_size) - 0.5

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden_layer = self.sigmoid(np.dot(inputs, self.input_weights))
        output_layer = self.sigmoid(np.dot(hidden_layer, self.output_weights))
        return np.argmax(output_layer)
    
    def copy_and_mutate(self, mutation_rate=MUTATION_RATE):
        new_nn = NeuralNetwork(input_size=self.input_weights.shape[0], 
                            hidden_size=self.input_weights.shape[1], 
                            output_size=self.output_weights.shape[1])
        
        new_nn.input_weights = self.input_weights + np.random.randn(*self.input_weights.shape) * mutation_rate
        new_nn.output_weights = self.output_weights + np.random.randn(*self.output_weights.shape) * mutation_rate
    
        return new_nn

class Bot:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.brain = brain or NeuralNetwork()
        self.is_dead = False
        self.has_finished = False
        self.steps = 0
        self.distance = 0
        self.fitness = 0
        self.repopulation_probability = 0

class Candy:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Game:
    def __init__(self):
        self.bots = [] 
        self.candy = False
    
    def start_new_game(self):
        self.bots = []
        for _ in range(NUM_BOTS):
            self.bots.append(Bot(STARTING_COORDS_BOT[0], STARTING_COORDS_BOT[1]))

        self.candy = Candy(STARTING_COORDS_CANDY[0], STARTING_COORDS_CANDY[1])

        for iteration in range(GAME_ITERATIONS):
            self.bots = self.run_game(iteration)

    def run_game(self, iteration):
        for _ in range(ROUNDS_PER_GAME):
            for bot in self.bots:
                if bot.is_dead or bot.has_finished:
                    continue

                board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
                board[bot.y][bot.x] = 1
                board[self.candy.y][self.candy.x] = 2

                direction = bot.brain.forward(board.flatten())
                if direction == 0:
                    bot.y += 1
                elif direction == 1:
                    bot.x += 1
                elif direction == 2:
                    bot.y -= 1
                elif direction == 3:
                    bot.x -= 1

                bot.steps += 1

                if bot.x == self.candy.x and bot.y == self.candy.y:
                    bot.has_finished = True

                if bot.y < 0 or bot.y >= BOARD_SIZE or bot.x < 0 or bot.x >= BOARD_SIZE:
                    bot.is_dead = True 
        
        total_distance = 0
        max_distance = 0
        bots_finished = 0
        total_steps = 0
        max_steps = 9999
        alive_bots = [bot for bot in self.bots if not bot.is_dead]

        for bot in alive_bots:
            bot.distance = math.sqrt((bot.x - self.candy.x)**2 + (bot.y - self.candy.y)**2)
            
            total_steps += bot.steps

            if bot.distance == 0:
                bot.finished = True
                bot.distance = 0.1 
                bots_finished += 1

                max_steps = max(max_steps, bot.steps)

            total_distance += bot.distance
            max_distance = max(max_distance, bot.distance)

        total_fitness = sum((max_distance / bot.distance) ** 2 * (1 + (max_steps - bot.steps) / max_steps) ** 3 for bot in alive_bots)
        avg_distance = total_distance / len(alive_bots)
        avg_steps = total_steps / len(alive_bots)

        for bot in alive_bots:
            bot.fitness = (max_distance / bot.distance) ** 2 * (1 + (max_steps - bot.steps) / max_steps) ** 3
            bot.repopulation_probability = bot.fitness / total_fitness

        new_bots = []
        for bot in self.bots:
            num_new_bots = custom_round(NUM_BOTS * bot.repopulation_probability)
            for _ in range(num_new_bots):
                new_bot = Bot(STARTING_COORDS_BOT[0], STARTING_COORDS_BOT[1], bot.brain)
                new_bot.brain = new_bot.brain.copy_and_mutate()
                new_bots.append(new_bot)

        print(f"Iter: {iteration}, Avg Distance: {avg_distance:.2f}, Alive: {len(alive_bots) / len(self.bots) * 100:.2f}%, Finished: {bots_finished / len(self.bots) * 100:.2f}%, Avg Steps: {avg_steps:.2f}")
        return new_bots

def custom_round(value):
    if value % 1 == 0.5:
        return math.ceil(value)
    else:
        return round(value)

if __name__ == "__main__":
    game = Game()
    game.start_new_game()
