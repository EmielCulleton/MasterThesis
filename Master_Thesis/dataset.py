import config
import torch


class EntityDataset:
    def __init__(self, texts, pos, tags):
        # texts: [["Hi", ",", "my", "name", "is", "Emiel"], ["Hello", "World"]]
        # pos/tags : [[1 2 3 4 1 6 ], [....], .....]
        self.texts = texts
        self.pos = pos
        self.tags = tags

    def  __init__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]      
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s, 
                add_special_tokens=False
            )

            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_pos.extend([tags[i]] * input_len)

        ids = ids[:config.MAX_LEN - 2] #-2 becaus some special tokens need to be added
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [101] + target_tag + [102]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)
        
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }       