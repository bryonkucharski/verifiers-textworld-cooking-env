# textworld_cooking_env

### Overview
- **Environment ID**: `textworld_cooking_env`
- **Short description**: Multi-turn text-based cooking game environment using TWCooking environment; rewards cumulative score across game steps.
- **Tags**: games, multi-turn, textworld, cooking, interactive-fiction, xml

### Datasets
- **Primary dataset(s)**: TWCooking dataset from GATA-public (auto-downloaded on first run)
- **Source links**: [GATA-public releases](https://github.com/xingdi-eric-yuan/GATA-public/releases)
- **Split sizes**: Train and test splits available at multiple difficulty levels (1-10)

### Difficulties 

Textworld cooking env is designed to have a progression of difficulties. As you progress, the agent is required to either include more items in the recipe, navigate more rooms, or prepare each item differently. The table below outlines exactly the difference between difficulities

| Difficulty | Recipe Size (Num Items)  | Num Locations | Max Score |  Need Cut |  Need Cook | 
| --- | ---- | ---- | ---- |---- |---- |
| 1 | 1 | 1 | 3 | ✗ | ✗ |
| 2 | 1 | 1 | 4 | ✗ | ✓ |
| 3 | 1 | 1 | 4 | ✓ | ✗ |
| 4 | 1 | 6 | 3 | ✗ | ✗ |
| 5 | 1 | 9 | 3 | ✗ | ✗ |
| 6 | 1 | 12 | 3 | ✗ | ✗ |
| 7 | 1 | 1 | 5 | ✓ | ✓ |
| 8 | 3 | 6 | 5 | ✗ | ✗ |
| 9 | 3 | 6 | 11 | ✓ | ✓ |
| 10 | 3 | 12 | 11 | ✓ | ✓ |

### Task
- **Type**: multi-turn (game interaction)
- **Parser**: `XMLParser` with `action` field
- **Rubric overview**: `EpisodicSumRubric` sums step rewards across the game trajectory

### How it works
- **Data preparation**: Downloads and extracts TWCooking game files to `~/.cache/textworld/tw-cooking/` on first run.
- **Game setup**: Each episode loads a `.z8` game file with TextWorld, providing observations, walkthroughs, and command templates.
- **Agent interaction**: The agent receives text observations and must provide actions in `<action></action>` XML tags (e.g., `<action>take knife from counter</action>`).
- **Scoring**: Rewards are computed as delta scores between steps; final reward is the sum of all step rewards.

### Quickstart

Example eval script

```bash
vf-eval -n -1 -s -v -k "<API key>"\
  -b "http://0.0.0.0:8000/v1"\
  -m "Qwen/Qwen3-4B-Thinking-2507"\ 
  --env-args '{"max_turns": 10,"difficulties": [1]}'\
  textworld_cooking_env
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulties` | List[int] | `[1]` | List of difficulty levels (1-10) to include in dataset |
| `system_prompt` | str | *(built-in)* | Custom system prompt for the agent |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Cumulative sum of step rewards (delta scores) across the episode |

