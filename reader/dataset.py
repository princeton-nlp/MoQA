from dataclasses import dataclass
from typing import Any, Dict, List

import torch

@dataclass
class ReaderDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = torch.tensor([d["input_ids"] for d in features], dtype=torch.long)
        attention_mask = torch.tensor([d["attention_mask"] for d in features], dtype=torch.long)
        offset_mapping = torch.tensor([d["offset_mapping"] for d in features], dtype=torch.long)
        start_positions = torch.tensor([d["start_positions"] for d in features], dtype=torch.long) if "start_positions" in features[0] else None
        end_positions = torch.tensor([d["end_positions"] for d in features], dtype=torch.long) if "end_positions" in features[0] else None
        answer_mask = torch.tensor([d["answer_mask"] for d in features], dtype=torch.long) if "answer_mask" in features[0] else None

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "answer_mask": answer_mask,
        }

        return output
