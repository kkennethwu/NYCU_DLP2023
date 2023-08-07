import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--config', is_config_file=True, help='Path to the configuration file.')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Script mode: train or test.')
    parser.add_argument('--ResnetModel', choices=['18', '50', '152'], default='train', help='Script mode: train or test.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=20, help='Epoches.')

    return parser