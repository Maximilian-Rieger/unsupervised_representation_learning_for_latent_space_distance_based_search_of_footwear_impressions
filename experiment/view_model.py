import argparse
import json
from utils.utils import load_config
from utils.utils import dynamic_import_experiment, make_dict
from torchview import draw_graph
from evaluation.experiments_overview import get_experiment_file
from experiment.data_zoo import DataZoo


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Create a graph of the model.')
    parser.add_argument('--log_dir', default=None, required=True, type=str, metavar='PATH', help='Path to logdir where experiment data are')
    parser.add_argument('--experiment', default=None, required=True, type=str, metavar='EXPERIMENT', help='Name of experiment')
    parser.add_argument('--save', default=True, type=bool, metavar='SAVE', help='Whether to save the graph or not')
    args = make_dict(parser.parse_args())
    experiment_file = get_experiment_file(args['log_dir'], args['experiment'], "experiment")
    options_file = get_experiment_file(args['log_dir'], args['experiment'], "options")
    options = {}
    with open(options_file, "r") as f:
        options = json.load(f)
    configArgs = load_config(options)
    experiment_mode = experiment_file.split("\\")[-1].split(".")[0]
    experiment = dynamic_import_experiment("experiment." + experiment_mode, configArgs)
    model = experiment.load_model()
    # check if options have input_channels and use it
    input_channels = None
    if 'channels' in options:
        input_channels = options['channels']
    elif 'model' in options and 'input_channels' in options['model']:
        input_channels = options['model']['input_channels']
    elif 'model' in options and 'in_chan' in options['model']:
        input_channels = options['model']['in_chan']
    else:
        input_channels = 1
    # check if options have input_size, image_size, img_size or input_shape and use it
    input_size = None
    if 'input_size' in options:
        input_size = options['input_size']
    elif 'image_size' in options:
        input_size = options['image_size']
    elif 'img_size' in options:
        input_size = options['img_size']
    elif 'input_shape' in options:
        input_size = options['input_shape']
    else:
        input_size = (128, 128)
    # check if input_size is a tuple, list or int and convert it to a tuple
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    # check if options have batchsize and use it
    batch_size = None
    if 'batchsize' in options:
        batch_size = options['batchsize']
    elif 'batch_size' in options:
        batch_size = options['batch_size']
    else:
        batch_size = 1
    print(model)
    # model_graph = draw_graph(model, input_size=(2, input_channels, *input_size), device='meta', save_graph=args['save'])
    # model_graph.visual_graph


if __name__ == '__main__':
    main()
