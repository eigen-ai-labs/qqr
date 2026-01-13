from qqr import registers
from qqr.schemas import GroupRewardModel, RewardModel


def get_reward_model(name: str) -> RewardModel | GroupRewardModel:
    model_key = name

    if "/" in model_key:
        model_key = name.split("/")[0]

    if model_key not in registers.reward_model:
        raise ValueError(
            f"Model not found for name '{name}' (parsed as '{model_key}'). "
            f"Available reward models: {list(registers.reward_model.keys)}"
        )

    return registers.reward_model[model_key]
