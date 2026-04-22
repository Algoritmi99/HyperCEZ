from hypercez import Hparams
from hypercez.envs.cl_env import CLEnvLoader


def main():
    env_name = "CW10"
    hparams = Hparams(env_name)

    hparams.add_ez_hparams(2)
    hparams.add_hnet_hparams()

    cl_env_loader = CLEnvLoader(hparams.env, seed=42)
    for i in range(hparams.num_tasks):
        cl_env_loader.add_task(i)

    env = cl_env_loader.get_env(0)

    print("observation space------------------------------")
    print(env.observation_space)
    print(len(env.observation_space.sample()))

    print("action space------------------------------")
    print(env.action_space)
    print(len(env.action_space.sample()))

    print("step return------------------------------")
    print(env.step(env.action_space.sample()))
    print(len(env.step(env.action_space.sample())))

    print("reset return------------------------------")
    print(len(env.reset()))


if __name__ == "__main__":
    main()