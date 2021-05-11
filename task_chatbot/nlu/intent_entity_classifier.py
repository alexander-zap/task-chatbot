import json
import re
from collections import namedtuple
from typing import Text

from dialogue_agent.dialog_config import slot_name_translations


class IntentEntityClassifier:
    def __init__(self, regex_file_path: Text):
        self.nlu_path = regex_file_path

    def __call__(self, utterance: Text):
        """
        :param utterance: Utterance for which intent and entities should be classified
        :return namedtuple("NLU_Response", "intent entities") if NLU classification is successful, else None
        """
        regex_entries = []
        with open(self.nlu_path, encoding="utf-8") as regex_nlu_file:
            regex_nlu = json.load(regex_nlu_file)
            for regex_entry in regex_nlu['patterns']:
                regex_entries.append((regex_entry['pattern'], regex_entry['intent'], regex_entry['entities']))

        nlu_response = namedtuple("NLU_Response", "intent entities")

        slot_translations = dict((k.lower(), v) for k, v in slot_name_translations.items())

        for pattern, intent, entity_keys in regex_entries:
            entity_dict = {}
            if re.match(pattern, utterance):
                match = re.search(pattern, utterance)
                groups = match.groups()
                for i in range(len(groups)):
                    entity_value = groups[i]
                    if "slot_name" in entity_keys and entity_value in slot_translations.keys():
                        entity_value = slot_translations[entity_value]
                    entity_dict.update({entity_keys[i]: entity_value})
                return nlu_response(intent, entity_dict)

        return None
