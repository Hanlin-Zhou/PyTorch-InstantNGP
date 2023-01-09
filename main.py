from config_parser import *
from grid import *
from decoder import *
from image_trainer import *
from nerf_trainer import *


def get_trainer(args):
    if args.task_type == "image":
        return ImageTrainer(args)
    elif args.task_type == "nerf":
        return NerfTrainer(args)


if __name__ == '__main__':
    args = parse_arg_from_yaml()
    setup_result_output(args)
    logging_setup(args.result_directory)
    my_trainer = get_trainer(args)
    my_trainer.train()
    print("Exiting...")


