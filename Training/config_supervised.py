import configargparse

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

    parser.add_argument('--seed', 
                        type=int, 
                        default=0, 
                        help='random seed (default: 123456)')

    parser.add_argument('--action_dim', 
                        type=int, 
                        default=4, 
                        help='action size')

    parser.add_argument('--inp_dim', 
                        type=int, 
                        default=12, 
                        help='state size')

    parser.add_argument('--log_steps', 
                        type=int, 
                        default=10, 
                        help='episodes before logging stats')

    parser.add_argument('--save_iter', 
                        type=int, 
                        default=1000, 
                        help='number of episodes until checkpoint is saved')

    parser.add_argument('--dt', 
                        type=float, 
                        default=0.001, 
                        help='dt of environment')

    parser.add_argument('--epochs', 
                        type=int, 
                        default=10000, 
                        help='maximum episodes to run')
    
    # Saving Parameters
    parser.add_argument('--model_save_path', 
                        type=str, 
                        default='',
                        help='path to folder and file name of model to save (do not put extension pth)')

    parser.add_argument('--batch_size', 
                        type=int, 
                        default=8, 
                        help='Size of sample from replay memory to update')

    return parser