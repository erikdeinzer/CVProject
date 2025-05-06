from imports import *

def load_config(filepath):
    config_dict = {}
    with open(filepath, 'r') as f:
        code = f.read()
    exec(code, {}, config_dict)
    return config_dict


def main(dataset_cfg: dict, 
         model_cfg: dict | list[dict],
         eval_cfg: dict,
         train_cfg,
         optim_cfg,):
    
    dataset = dataset_builder.build_module(**dataset_cfg)

    if isinstance(model_cfg, dict):
        model = model_builder.build_module(**model_cfg)
    elif isinstance(model_cfg, list):
        model = nn.Sequential([
            model_builder.build_module(**cfg)
            for cfg in model_cfg
        ])
    else:
        raise TypeError('Model config must either be a dict or a list of dicts')
    



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










