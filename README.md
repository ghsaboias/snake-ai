# Project Name

## Description

This project is a Snake Game AI implemented in Python. It utilizes a deep learning model to learn and make decisions based on the game state. The AI agent is trained using a Q-learning algorithm, which is a type of reinforcement learning.

The AI agent is defined in `scripts/agent.py`, and the game logic is implemented in `scripts/game.py`. The deep learning model and its training logic are in `scripts/model.py`. The `scripts/helper.py` file contains utility functions for plotting the training progress.

## Project Structure

```
.
├── scripts/
│   ├── agent.py
│   ├── game.py
│   ├── helper.py
│   └── model.py
├── model/
│   └── model.pth
├── requirements.txt
├── training.png
├── arial.ttf
└── README.md
```

- `scripts/agent.py`: Contains the `Agent` class responsible for making decisions based on the game state and learning from its actions.
- `scripts/game.py`: Contains the `SnakeGameAI` class which implements the game logic.
- `scripts/helper.py`: Contains utility functions for plotting the training progress.
- `scripts/model.py`: Contains the `Linear_QNET` class which is the deep learning model used by the agent, and the `QTrainer` class responsible for training the model.
- `model/model.pth`: The trained model weights.

## Running the Project

1. Clone the repository:

```bash
git clone git@github.com:ghsaboias/snake-ai.git
cd snake-ai
```

2. Create a virtual environment:

```bash
python -m venv pygame_env
```

3. Activate the virtual environment:

- For Windows:

```bash
pygame_env\Scripts\activate
```

- For macOS and Linux:

```bash
source pygame_env/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Agent:

```bash
python scripts/agent.py
```

6. Check the training progress by opening the `training.png` file.

## Troubleshooting

- High CPU Usage: If you experience high CPU usage or a 'trace trap' error, optimize your `scripts/helper.py` functions and avoid excessive plotting in tight loops.

## Contributing

If you wish to contribute to this project, please fork the repository and create a pull request with your changes. Make sure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
