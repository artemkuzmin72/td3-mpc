from setuptools import setup, find_packages

package_name = 'turtlebot_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/display.launch.py']),
        ('share/' + package_name + '/urdf', ['urdf/robot.urdf.xacro']),
        ('share/' + package_name + '/weights', ['weights/actor.pth'])
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
            'sim_base = turtlebot_rl.sim_base:main',
        ],
    },
)
