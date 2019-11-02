import random
from collections import deque

import numpy as np
import pygame
from tensorflow.python import keras
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense

from src.input_manager import KeyEventType, EventHandler, InputManager
from src.player import Game, Agent, ActionHandler

number_of_iterations_per_play = None


class Model:
    __batch_size = 100

    def __init__(self):
        self.model = Sequential([Dense(40, input_shape=(4,), activation="relu"),
                                 Dense(40, activation="relu"),
                                 Dense(40, activation="relu"),
                                 Dense(4, activation="tanh")])
        # TODO: xavier initialization?
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.Adam(lr=0.001))

        self.memory = deque(maxlen=10000)

    def experience_replay(self):
        if len(self.memory) < self.__batch_size:
            return

        batch = random.sample(self.memory, self.__batch_size)
        states = np.vstack([state for state, _, _, _, _ in batch])
        next_states = np.vstack([next_state for _, _, _, next_state, _ in batch])

        predicted_states = self.model.predict(states)
        predicted_next_states = self.model.predict(next_states)
        max_nex_state_values = np.max(predicted_next_states, 1)

        for index, (_, action, reward, _, terminal) in enumerate(batch):
            q_update = reward

            if not terminal:
                discount_factor = 0.95
                q_update += discount_factor * max_nex_state_values[index]

            learning_rate = 0.95
            predicted_states[index][action] = ((1 - learning_rate) * predicted_states[index][action]
                                               + learning_rate * q_update)
        self.model.fit(states, predicted_states, verbose=0)


if __name__ == "__main__":
    model = Model()

    player = Agent("player.png")
    player.randomize_position()

    computers = [Agent("computer.png")]

    for computer in computers:
        computer.randomize_position()

    input_manager = InputManager()
    input_manager.add_event_handler(EventHandler(pygame.K_LEFT, KeyEventType.CONTINUOUSLY_PRESSED,
                                                 lambda: player.move(np.array([-10, 0]))))
    input_manager.add_event_handler(EventHandler(pygame.K_RIGHT, KeyEventType.CONTINUOUSLY_PRESSED,
                                                 lambda: player.move(np.array([10, 0]))))
    input_manager.add_event_handler(EventHandler(pygame.K_UP, KeyEventType.CONTINUOUSLY_PRESSED,
                                                 lambda: player.move(np.array([0, -10]))))
    input_manager.add_event_handler(EventHandler(pygame.K_DOWN, KeyEventType.CONTINUOUSLY_PRESSED,
                                                 lambda: player.move(np.array([0, 10]))))
    action_handler = ActionHandler(model)
    game = Game(player, computers, input_manager, action_handler)

    number_of_tries = [(0, 1), (1000, 0.9), (2000, 0.7), (3500, 0.5), (5000, 0.3), (6000, 0.1)]
    i = 0

    qqq = 0
    while True:
        qqq += 1
        if qqq % 10 == 0:
            model.model.save_weights("./model/my_model")
            print("saving weights")

        print("resetted")
        game.reset()
        number_of_iterations_per_play = 0
        while True:
            i += 1
            number_of_iterations_per_play += 1

            should_action_be_random = False

            for try_number, percent in reversed(number_of_tries):
                if i > try_number:
                    if random.random() <= percent:
                        should_action_be_random = True
                    break

            terminal = game.single_iteration(should_action_be_random)

            if i % 100 == 0:
                print("local iteration:", qqq, "global iteration:", i)

            model.experience_replay()

            if np.any(terminal):
                break

    # model.model.load_weights("./model/my_model")
    #
    # while True:
    #     done = game.single_iteration(False)
    #
    #     if np.any(done):
    #         game.reset()
