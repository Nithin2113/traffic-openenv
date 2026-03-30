from app.env import TrafficEnv

env = TrafficEnv()
state = env.reset()
print("Initial:", state)

for i in range(5):
    state, reward, done, _ = env.step(1)
    print(f"Step {i+1}:", state, reward.value)