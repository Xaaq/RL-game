from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

import pygame


class KeyEventType(Enum):
    JUST_PRESSED = 0
    CONTINUOUSLY_PRESSED = 1
    JUST_RELEASED = 2


@dataclass
class EventHandler:
    key: int
    event_type: KeyEventType
    function_to_execute: Callable[[], None]


# noinspection PyUnresolvedReferences
class InputManager:
    def __init__(self):
        self.__key_presses_states: Dict[int, bool] = {}
        self.__event_handlers: List[EventHandler] = []

    def add_event_handler(self, event_handler: EventHandler):
        self.__event_handlers.append(event_handler)

        if (event_handler.event_type == KeyEventType.CONTINUOUSLY_PRESSED
                and event_handler.key not in self.__key_presses_states):
            self.__key_presses_states[event_handler.key] = False

    def execute_input_calls(self):
        for input_event in pygame.event.get():
            if input_event.type == pygame.KEYDOWN and input_event.key in self.__key_presses_states:
                self.__key_presses_states[input_event.key] = True
            elif input_event.type == pygame.KEYUP and input_event.key in self.__key_presses_states:
                self.__key_presses_states[input_event.key] = False

            for event_handler in self.__event_handlers:
                are_pressed_conditions_met = (input_event.type == pygame.KEYDOWN
                                              and event_handler.event_type == KeyEventType.JUST_PRESSED)
                are_released_condition_met = (input_event.type == pygame.KEYUP
                                              and event_handler.event_type == KeyEventType.JUST_RELEASED)
                if (are_pressed_conditions_met or are_released_condition_met) and event_handler.key == input_event.key:
                    event_handler.function_to_execute()

        for event_handler in self.__event_handlers:
            if (event_handler.event_type == KeyEventType.CONTINUOUSLY_PRESSED
                    and self.__key_presses_states[event_handler.key] is True):
                event_handler.function_to_execute()
