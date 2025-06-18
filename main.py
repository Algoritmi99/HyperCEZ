from hypercez.hypernet.hypercl import MLP


def main():
    mlp = MLP()
    print(mlp.param_shapes)

    print([list(i.shape) for i in mlp.parameters()])

if __name__ == "__main__":
    main()
