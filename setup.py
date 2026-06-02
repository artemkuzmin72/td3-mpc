from setuptools import setup, find_packages


setup(
    name='turtlebot_rl',
    version='0.0.0',
    packages=find_packages(),

    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/turtlebot_rl']),
    ('share/turtlebot_rl', ['package.xml']),
    ('share/turtlebot_rl/launch', ['launch/maze.launch.py']),
    ('share/turtlebot_rl/worlds', ['worlds/maze.world']),
],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artem',
    maintainer_email='frankiekoo72@gmail.com',
    description='RL robot package',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'rl_node = turtlebot_rl.rl_node:main',
            'env_node = turtlebot_rl.env_node:main',
            'train_node = turtlebot_rl.train.train_node:main',
            'sim_node = turtlebot_rl.sim_node:main'
        ],
    },
)
