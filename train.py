from imports import *

required_config_keys = [
    'train_dataloader',
    'model',
    'optimizer',
    'lr_scheduler',]


def load_config(filepath):
    config_dict = {}
    with open(filepath, 'r') as f:
        code = f.read()
    exec(code, {}, config_dict)
    # Check if all required keys are present in the config
    for key in required_config_keys:
        if key not in config_dict:
            raise ValueError(f"Missing required config key: {key}")
    return config_dict




def main(config):


    if isinstance(config['model'], dict):
        model = model_builder.build_module(**config['model'])
    elif isinstance(config['model'], list):
        model = nn.Sequential([
            model_builder.build_module(**cfg)
            for cfg in config['model']
        ])
    else:
        raise TypeError('Model config must either be a dict or a list of dicts')
    
    train_dataloader = data.build_dataloader_from_cfg(config['train_dataloader'])
    
    optimizer = utils.build_optimizer(model, config['optimizer'])





# ------------------------------------------------------------- #
# ------------------- Define Input Arguments ------------------ #
# ------------------------------------------------------------- #


# Define args
import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
parser.add_argument('--work-dir', type=str, default='./work_dirs', help='Path to work directory')
parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint file to resume from')
args = parser.parse_args()


# Read provided config when running the script with the config arg
if args.config:
    cfg = load_config(args.config)
else:
    raise ValueError('No config file provided. Please provide a config file with --config argument.')

# Set work directory and create it if it doesn't exist
os.makedirs(args.work_dir, exist_ok=True)
t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
work_dir = os.path.join(args.work_dir, t)
os.makedirs(work_dir, exist_ok=True)
print(f'Work directory: {work_dir}')
cfg['work_dir'] = work_dir

# Set resume_from if provided
if args.resume_from:
    if not os.path.exists(args.resume_from):
        raise ValueError(f'Resume file {args.resume_from} does not exist.')
    else:
        print(f'Resuming from {args.resume_from}')
        cfg['resume_from'] = args.resume_from

# Save the config file to the work directory
with open(os.path.join(work_dir, 'config.py'), 'w') as f:
    f.write(cfg)


# Save the command line arguments to the work directory
with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
    f.write(str(args))



# -------------------------------------------------------------- #
# ----------------- Execute with parsed Config ----------------- #
# -------------------------------------------------------------- #

main(**cfg)










