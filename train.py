import argparse


def main():
    # Arguments
    preparser = argparse.ArgumentParser(description='Training model.')
    preparser.add_argument('--type', type=str, help='type of training ')

    # parse known args and keep the remaining args to pass into sub-parser
    paras, remaining = preparser.parse_known_args()
    if paras.type == "synth":
        from control.cli.synthesizer_train import new_train
        new_train(remaining)
    if paras.type == "vits":
        from models.synthesizer.train_vits import new_train
        new_train(remaining)


if __name__ == "__main__":
    main()
