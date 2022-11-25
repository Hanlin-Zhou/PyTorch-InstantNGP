from config_parser import *
from grid import *
from decoder import *
from trainer import *


if __name__ == '__main__':
    args = parse_arg_from_yaml()
    setup_result_output(args)
    logging_setup(args.result_directory)
    my_grid = Grid(args.feature_dim, args.grid_dim, args.num_lvl, args.max_res, args.min_res, args.hashtable_power,
                   args.force_cpu)
    my_decoder = Decoder(args.input_dim, args.output_dim, args.activation, args.last_activation, args.bias,
                         args.num_layer, args.hidden_dim)
    my_trainer = Trainer(args.num_epoch, args.batch_size, args.range_clamping, args.save_every, args.log_every,
                         args.learning_rate, (args.beta1, args.beta2), args.eps, args.weight_decay, my_grid, my_decoder,
                         args.result_directory, args.source_directory, args.force_cpu)
    my_trainer.train()
    print("Exiting...")

