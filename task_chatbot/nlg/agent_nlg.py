from typing import Text

import dialogue_agent.dialog_config as dia_config
from dialogue_agent.action import AgentAction


class AgentNLG:
    def __call__(self, agent_action: AgentAction) -> Text:
        """
        Transform a structured agent action into a natural language utterance
        :param agent_action: agent action [instance of AgentAction class]
        :return: utterance [Text] generated from structured agent action
        """

        def translate(slot_name):
            """ Translates slot_name to German by slot_name_translations lookup."""
            reverted_slot_name_translations = {v: k for k, v in dia_config.config.slot_name_translations.items()}
            if slot_name in reverted_slot_name_translations:
                slot_name = reverted_slot_name_translations[slot_name]
            return slot_name

        agent_utterance = ""

        if agent_action.intent == 'inform' or agent_action.intent == 'match_found':
            inform_slot_key = translate(list(agent_action.inform_slots.keys())[0])
            inform_slot_value = list(agent_action.inform_slots.values())[0]
            if agent_action.intent == 'inform':
                if inform_slot_value == 'no match available':
                    agent_utterance = "Ich konnte für {} leider keinen Match finden.".format(inform_slot_key)
                else:
                    agent_utterance = "Als {} wäre {} möglich.".format(inform_slot_key, inform_slot_value)
            elif agent_action.intent == 'match_found':
                if 'no match available' in inform_slot_value:
                    agent_utterance = "Ich konnte leider kein passendes Ticket finden."
                else:
                    match_string = "\n".join(
                        [f"{slot}: {value}" for slot, value in agent_action.inform_slots.items()]
                    )
                    agent_utterance = f"Kann ich Ihnen folgendes Ticket empfehlen?\n{match_string}"
        elif agent_action.intent == 'request':
            request_slot_key = translate(agent_action.request_slots[0])
            agent_utterance = "Was wünschen Sie als {}?".format(request_slot_key)
        elif agent_action.intent == 'done':
            agent_utterance = "Auf Wiedersehen."

        return agent_utterance
