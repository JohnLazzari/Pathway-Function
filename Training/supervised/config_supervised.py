import configargparse
import argparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-3, 
                        help='learning rate (default: 0.001)')

    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=1e-3, 
                        help='weight decay parameter')

    parser.add_argument('--noise_level_act', 
                        type=float, 
                        default=0.15, 
                        help='activation noise')

    parser.add_argument('--noise_level_inp', 
                        type=float, 
                        default=0.01, 
                        help='input noise')

    parser.add_argument('--activation_name', 
                        type=str, 
                        default="relu", 
                        help='act')
    
    parser.add_argument('--constrained', action=argparse.BooleanOptionalAction)

    parser.add_argument('--device', 
                        type=str, 
                        default="cpu", 
                        help='device')
    
    parser.add_argument('--dt', 
                        type=float, 
                        default=10, 
                        help="timestep dt")

    parser.add_argument('--t_const', 
                        type=float, 
                        default=100, 
                        help='tau for network neurons')

    parser.add_argument('--batch_first', action=argparse.BooleanOptionalAction)

    parser.add_argument('--seed', 
                        type=int, 
                        default=123456, 
                        help='random seed (default: 123456)')

    parser.add_argument('--save_iter', 
                        type=int, 
                        default=100, 
                        help='number of episodes until checkpoint is saved')

    parser.add_argument('--epochs', 
                        type=int, 
                        default=10000, 
                        help='maximum episodes to run')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=1, 
                        help='Size of sample from replay memory to update')

    # Saving Parameters
    parser.add_argument('--model_save_path', 
                        type=str, 
                        default='checkpoints/mRNN.pth',
                        help='path to folder and file name of model to save (do not put extension pth)')

    parser.add_argument('--model_config_path', 
                        type=str, 
                        default='Training/configurations/mRNN_thal_inp.json',
                        help='path to folder and file name of model to save (do not put extension pth)')

    return parser