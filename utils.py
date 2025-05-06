import torch

def build_optimizer(model, cfg):
    """
    Builds an optimizer for the given model.

    Args:
        model (nn.Module): The model whose parameters the optimizer will update.
        cfg (dict): Optimizer config with 'type' and 'args' keys.

    Returns:
        torch.optim.Optimizer
    """
    if not isinstance(cfg, dict):
        raise TypeError('Optimizer config must be a dict')

    opt_type = cfg['type']
    opt_args = cfg.get('args', {})

    if not hasattr(torch.optim, opt_type):
        raise ValueError(f"Optimizer '{opt_type}' is not available in torch.optim")

    return getattr(torch.optim, opt_type)(model.parameters(), **opt_args)
