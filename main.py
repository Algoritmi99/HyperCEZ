from hypercez import Hparams


def main():
    params = Hparams("pusher")
    params.add_hnet_hparams()
    print(params.lr)


if __name__ == "__main__":
    main()
