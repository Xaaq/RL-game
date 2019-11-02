from typing import List

import numpy as np
import pygame

from src.input_manager import InputManager


class Globals:
    screen_width = 1000
    screen_height = 800


class Agent:
    def __init__(self, image_name: str):
        self.position = np.array([0, 0])
        self.__image = pygame.image.load(image_name)

    def move(self, delta):
        new_position = self.position + delta
        self.position = np.clip(new_position, [0, 0], self.__get_position_max_bounds())

    def draw(self, screen):
        screen.blit(self.__image, self.position)

    def randomize_position(self):
        self.position = np.random.random(2) * self.__get_position_max_bounds()

    def get_state(self, agent: "Agent"):
        concatenated_positions = np.hstack((self.position, agent.position))
        normalized_positions = concatenated_positions / np.array([Globals.screen_width, Globals.screen_height,
                                                                  Globals.screen_width, Globals.screen_height])
        return normalized_positions * 2 - 1

    def __get_position_max_bounds(self):
        image_width, image_height = self.__image.get_rect().size
        position_max_bounds = [Globals.screen_width - image_width, Globals.screen_height - image_height]
        return position_max_bounds


class ActionHandler:
    def __init__(self, model):
        self.__model = model

    def execute_actions(self, player: Agent, agent_list: List[Agent], should_action_be_random: bool):
        states_before_action = self.get_agent_states(player, agent_list)

        if should_action_be_random:
            agent_chosen_actions = np.random.randint(0, 4, len(agent_list))
        else:
            agent_action_probabilities = self.__model.model.predict(states_before_action)
            agent_chosen_actions = agent_action_probabilities.argmax(1)

        for action, agent in zip(agent_chosen_actions, agent_list):
            if action == 0:
                agent.move(np.array([0, -50]))
            elif action == 1:
                agent.move(np.array([50, 0]))
            elif action == 2:
                agent.move(np.array([0, 50]))
            elif action == 3:
                agent.move(np.array([-50, 0]))

        states_after_action = self.get_agent_states(player, agent_list)

        agent_positions = np.array([agent.position for agent in agent_list])
        positions_difference = player.position - agent_positions

        done = np.hypot(positions_difference[:, 0], positions_difference[:, 1]) < 100
        rewards = np.where(done, 1, -0.01)

        concatenated_data = zip(states_before_action, agent_chosen_actions, rewards, states_after_action, done)

        for state_before, action, reward, state_after, is_done in concatenated_data:
            self.__model.memory.append((state_before, agent_chosen_actions[0], reward, state_after, is_done))

        return done

    @staticmethod
    def get_agent_states(player: Agent, agent_list: List[Agent]) -> np.ndarray:
        all_states = [agent.get_state(player) for agent in agent_list]
        return np.array(all_states)


class Game:
    def __init__(self, player: Agent, computers: List[Agent], input_manager: InputManager,
                 action_handler: ActionHandler):
        self.__screen = pygame.display.set_mode((Globals.screen_width, Globals.screen_height))
        self.__player = player
        self.__computers = computers
        self.__input_manager = input_manager
        self.__action_handler = action_handler

    def reset(self):
        self.__player.randomize_position()

        for computer in self.__computers:
            computer.randomize_position()

    def single_iteration(self, should_action_be_random: bool):
        self.__input_manager.execute_input_calls()

        done = self.__action_handler.execute_actions(self.__player, self.__computers, should_action_be_random)

        self.__draw_scene()
        return done

    def __draw_scene(self):
        self.__screen.fill((103, 58, 183))

        self.__player.draw(self.__screen)

        for computer in self.__computers:
            computer.draw(self.__screen)

        pygame.display.flip()
