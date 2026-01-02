import logging
import time
from typing import Annotated, Any, List

import gymnasium as gym
import textworld
import verifiers as vf
from annotated_types import Ge, Le
from datasets import Dataset
from textworld.envs.wrappers import Filter
from verifiers.rubrics.rubric import Rubric

from data import get_cooking_game, prepare_twcooking_data

DEFAULT_SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game wrapped in <action></action> XML tags e.g. <action>go north</action>"
    " When stuck, try using the `help` command to see what commands are available."
    "You must fill in only one the following templates to successfully make a move e.g. 'take {o} from {c}' a filled in action would be 'take knife from counter' "
)

tw_intro_text = """\n\n\n                    ________  ________  __    __  ________        \n                   |        \\|        \\|  \\  |  \\|        \\       \n                    \\$$$$$$$$| $$$$$$$$| $$  | $$ \\$$$$$$$$       \n                      | $$   | $$__     \\$$\\/  $$   | $$          \n                      | $$   | $$  \\     >$$  $$    | $$          \n                      | $$   | $$$$$    /  $$$$\\    | $$          \n                      | $$   | $$_____ |  $$ \\$$\\   | $$          \n                      | $$   | $$     \\| $$  | $$   | $$          \n                       \\$$    \\$$$$$$$$ \\$$   \\$$    \\$$          \n              __       __   ______   _______   __        _______  \n             |  \\  _  |  \\ /      \\ |       \\ |  \\      |       \\ \n             | $$ / \\ | $$|  $$$$$$\\| $$$$$$$\\| $$      | $$$$$$$\\\n             | $$/  $\\| $$| $$  | $$| $$__| $$| $$      | $$  | $$\n             | $$  $$$\\ $$| $$  | $$| $$    $$| $$      | $$  | $$\n             | $$ $$\\$$\\$$| $$  | $$| $$$$$$$\\| $$      | $$  | $$\n             | $$$$  \\$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$\n             | $$$    \\$$$ \\$$    $$| $$  | $$| $$     \\| $$    $$\n              \\$$      \\$$  \\$$$$$$  \\$$   \\$$ \\$$$$$$$$ \\$$$$$$$ \n\n"""

# prompt_template = """
# # Exact commands to run to beat the game

# {walkthrough}

# # Observation

# {obs}

# # Action Templates

# {command_templates}
# """

prompt_template = """

# Observation

{obs}

# Available commands templates: 
look:                describe the current room           
goal:                print the goal of this game  
inventory:           print player's inventory
go <dir>:            move the player north, east, south or west
examine ...:         examine something more closely   
eat ...:             eat edible food
open ...:            open a door or a container 
close ...:           close a door or a container 
drop ...:            drop an object on the floor     
take ...:            take an object that is on the floor  
put ... on ...:      place an object on a supporter     
take ... from ...:   take an object from a container or a supporter
insert ... into ...: place an object into a container                   
lock ... with ...:   lock a door or a container with a key            
unlock ... with ...: unlock a door or a container with a key      
cook ... with ...:   cook cookable food with something providing heat
slice ... with ...:  slice cuttable food with something sharp
chop ... with ...:   chop cuttable food with something sharp  
dice ... with ...:   dice cuttable food with something sharp 
prepare meal:        combine ingredients from inventory into a meal
"""



logger = logging.getLogger("verifiers.textworld_cooking_env")


class TextWorldEnv(gym.Env):
    def __init__(self, gamefile, admissible_commands=False, *args, **kwargs):
        self.infos = textworld.EnvInfos(
            score=True,
            max_score=True,
            won=True,
            lost=True,
            feedback=True,
            moves=True,
            command_templates=True,
            admissible_commands=admissible_commands,
            extras=["walkthrough"],
        )
        self.gamefile = gamefile
        self.env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_score = 0  # Initialize last_score at the start of each episode
        self.total_score=0
        
        if self.env is None:
            self.env = textworld.start(self.gamefile, self.infos, wrappers=[Filter])

        obs, infos = self.env.reset()
        self.max_score = infos['max_score']

        return obs, infos

    def step(self, action):
        observation, score, done, info = self.env.step(action)
       
        if done:
            if score != self.max_score:
                reward = -self.max_score
            else:
                reward = self.max_score

        else:
            # Calculate reward (delta score) for this step
            reward = score - self.last_score

        self.last_score = score
        
        return observation, reward, done, info


class TWCookingEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        difficulties: List[Annotated[int, Ge(1), Le(10)]] = [1],
        **kwargs,
    ):
        self.difficulties = difficulties

        prepare_twcooking_data(force=False)

        # each row in this dataset is one game_file
        dataset, eval_dataset = self.tw_to_hf()

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def tw_to_hf(self) -> tuple[Dataset, Dataset]:
        all_train_games = []
        for diff in self.difficulties:
            all_train_games.extend(sorted(get_cooking_game(diff, split="train")))

        all_eval_games = []
        for diff in self.difficulties:
            all_games = sorted(get_cooking_game(diff, split="test"))
            all_eval_games.extend(all_games)

        dataset_rows = []
        for game_file in all_train_games:
            env = TextWorldEnv(game_file)
            obs, infos = env.reset()

            command_templates = "\n".join(infos["command_templates"])
            question = prompt_template.format(
                obs=obs.replace(tw_intro_text, "").strip(),
                walkthrough=infos["extra.walkthrough"],
                command_templates=command_templates,
            )
            dataset_rows.append({"question": str(question), "answer": str(game_file)})

        eval_dataset_rows = []
        for game_file in all_eval_games:
            env = TextWorldEnv(game_file)
            obs, infos = env.reset()
            command_templates = "\n".join(infos["command_templates"])
            question = prompt_template.format(
                obs=obs.replace(tw_intro_text, "").strip(),
                walkthrough=infos["extra.walkthrough"],
                command_templates=command_templates,
            )
            eval_dataset_rows.append(
                {"question": str(question), "answer": str(game_file)}
            )

        dataset = Dataset.from_list(dataset_rows)
        eval_dataset = Dataset.from_list(eval_dataset_rows)

        return dataset, eval_dataset

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Setup the TextWorld environment."""
        s = time.time()
        game_file = state["answer"]
        state["tw_env"] = TextWorldEnv(game_file)
        state["tw_env"].reset()
        logger.debug(f"Running game file:  {game_file}")
        return state

    @vf.cleanup
    async def cleanup_tw_env(self, state: vf.State):
        state.pop("tw_env")

    @vf.stop
    async def game_completed(self, state: vf.State) -> bool:
        return state.get("game_completed", False)

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        env = state["tw_env"]

        # parse guess
        action = self.parser.parse_answer(messages)
        logger.debug(f"Parsed {action=}")

        # step env
        obs, score, done, infos = env.step(str(action))
        state["trajectory"][-1]["reward"] = score

        if done:
            logger.debug("Game completed!")
            state["game_completed"] = True

        return [{"role": "user", "content": obs}]


class EpisodicSumRubric(Rubric):
    def __init__(self, weight: float = 1.0, **kwargs: Any):
        super().__init__(
            funcs=[
                lambda state: float(
                    sum(
                        float(step.get("reward", 0.0) or 0.0)
                        for step in state.get("trajectory", [])
                    )
                )
            ],
            weights=[weight],
            **kwargs,
        )


### environment loader
def load_environment(
    difficulties: List[Annotated[int, Ge(1), Le(10)]] = [1],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    **kwargs,
):
    parser = vf.XMLParser(fields=["action"], answer_field="action")
    rubric = EpisodicSumRubric(parser=parser)

    return TWCookingEnv(
        difficulties,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
