# TurtleBot3 Model Predict Control
Architecture:
TD3 -> MPC -> TurtleBot3

Technologies:
- ROS2 Humble
- Gazebo
- TurtleBot3
- PyTorch
- TD3
- MPC

## Installation 

1.  Клонируйте репозиторий:
    ```bash
    git clone https://github.com/artemkuzmin72/td3-mpc
    cd td3-mpc
    ```
2.  Создайте и активируйте виртуальное окружение:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Установите необходимые библиотеки:
    ```bash
    pip3 install -r requirements.txt
    ```

## Running

1.  Запустите мир в Gazebo:
    ```bash
    ros2 launch turtlebot_rl maze.launch.py
    ```
2.  Запустите обучение:
    ```bash
    ros2 run turtlebot_rl train_node
    ```

## Weights

Веса модели автоматически сохраняются в папку weights, также сохраняется replay_buffer для последующего fine tuning.