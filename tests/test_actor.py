from torchinfo import summary

from td3_agent.actor import Actor


model = Actor(
    state_dim=24,
    action_dim=2,
    max_action=1.0
)

summary(
    model,
    input_size=(1, 24)
)
