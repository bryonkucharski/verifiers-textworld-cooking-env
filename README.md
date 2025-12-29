# textworld_cooking_env

### Overview
- **Environment ID**: `textworld_cooking_env`
- **Short description**: Multi-turn text-based cooking game environment using TWCooking environment; rewards cumulative score across game steps.
- **Tags**: games, multi-turn, textworld, cooking, interactive-fiction, xml

### Datasets
- **Primary dataset(s)**: TWCooking dataset from GATA-public (auto-downloaded on first run)
- **Source links**: [GATA-public releases](https://github.com/xingdi-eric-yuan/GATA-public/releases)
- **Split sizes**: Train and test splits available at multiple difficulty levels (1-10)

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
Run an evaluation with default settings:

```bash
uv run vf-eval textworld_cooking_env
```

Configure model and sampling:

```bash
uv run vf-eval textworld_cooking_env \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"difficulties": [1, 2, 3]}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The first run downloads the TWCooking dataset (~200MB) and may take a few minutes.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulties` | List[int] | `[1]` | List of difficulty levels (1-10) to include in dataset |
| `system_prompt` | str | *(built-in)* | Custom system prompt for the agent |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Cumulative sum of step rewards (delta scores) across the episode |

