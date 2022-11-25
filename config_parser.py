import sys
import argparse
import yaml


def get_parser():
    parser = argparse.ArgumentParser(description='ArgParser for InstantNGP Implementation CSC419')

    file_group = parser.add_argument_group('File')
    file_group.add_argument('--result_directory', type=str, help='path to result folder')
    file_group.add_argument('--source_directory', type=str, help='path to source')

    grid_group = parser.add_argument_group('Grid')
    grid_group.add_argument('--feature_dim', type=int, help='number feature dimension')
    grid_group.add_argument('--grid_dim', type=int, help='number of grid dimension')
    grid_group.add_argument('--num_lvl', type=int, help='number of grid levels')
    grid_group.add_argument('--max_res', type=int, help='max resolution')
    grid_group.add_argument('--min_res', type=int, help='min resolution')
    grid_group.add_argument('--hashtable_power', type=int, help='exponential of hashtable')

    decoder_group = parser.add_argument_group('Decoder')
    decoder_group.add_argument('--input_dim', type=int, help='number input dimension')
    decoder_group.add_argument('--output_dim', type=int, help='number output dimension')
    decoder_group.add_argument('--activation', type=str, help='which activation function')
    decoder_group.add_argument('--last_activation', type=str, help='which activation function for out layer')
    decoder_group.add_argument('--bias', type=bool, help='should use bias')
    decoder_group.add_argument('--num_layer', type=str, help='how many hidden layers')
    decoder_group.add_argument('--hidden_dim', type=str, help='width of hidden layer')

    trainer_group = parser.add_argument_group('Trainer')
    trainer_group.add_argument('--num_epoch', type=int, help='number of epoch')
    trainer_group.add_argument('--batch_size', type=int, help='batch size')
    trainer_group.add_argument('--range_clamping', type=bool, help='Do training in [0, 1] space or [0, 255] space')
    trainer_group.add_argument('--save_every', type=int, help='save every how many epoch')
    trainer_group.add_argument('--log_every', type=int, help='log every how many epoch')
    trainer_group.add_argument('--learning_rate', type=float, help='learning rate of Adam')
    trainer_group.add_argument('--beta1', type=tuple, help='beta1 for Adam')
    trainer_group.add_argument('--beta2', type=tuple, help='beta2 for Adam')
    trainer_group.add_argument('--eps', type=float, help='eps for Adam')
    trainer_group.add_argument('--weight_decay', type=float, help='weight regularization decay for decoder')
    trainer_group.add_argument('--force_cpu', type=bool, help='train on cpu')

    return parser


def parser_from_yaml(parser, yaml_path):
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    field_list = []
    for group in parser._action_groups:
        for action in group._group_actions:
            field_list.append(action.dest)
    field_list = set(field_list)

    default_dict = {}

    for key in config_dict:
        for field in config_dict[key]:
            if field not in field_list:
                raise ValueError(
                    f"ERROR: field {field} is not valid"
                )
            default_dict[field] = config_dict[key][field]

    parser.set_defaults(**default_dict)


def parse_arg_from_yaml(yaml_path='./config.yaml'):
    if len(sys.argv) == 1:
        yaml_path = './config.yaml'
    elif len(sys.argv) == 2:
        yaml_path = sys.argv[1]
    else:
        print("usage: main.py [path to yaml config file]")
        quit()
    parser = get_parser()
    parser_from_yaml(parser, yaml_path)
    args = parser.parse_args()
    args.config_path = yaml_path
    return args
