

def main():
    import torch
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 5)
            self.conv = nn.Conv2d(1, 3, kernel_size=3)
            self.bn = nn.BatchNorm1d(5)

    model = MyModel()
    state = model.state_dict()

    for k, v in state.items():
        print(k, v.shape)
        print(type(k))

if __name__ == "__main__":
    main()
