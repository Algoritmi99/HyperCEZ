import gymnasium as gym
import metaworld


def main():
    env = gym.make("Meta-World/MT1", env_name="reach-v3")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs)
    print(reward)
    print(terminated)
    print(truncated)
    print(info)
    print(env.observation_space)


if __name__ == "__main__":
    main()