class Registry():
    def __init__(self, name='default'):
        self.name = name
        self._registry = {}

    def register(self, obj=None, *, type=None):
        if obj is None:
            return lambda obj: self.register(obj, type=type)
        key = type or obj.__name__
        if key in self._registry:
            raise ValueError('Duplicate registrations with same name not possible')
        self._registry[key]=obj
        return obj
    
    def get(self, type):
        if type not in self._registry:
            raise KeyError(f"'{type}' not found in the '{self.name}' registry.")
        obj = self._registry.get(type)
        return obj
    
    def all(self):
        return self._registry
    
    def __contains__(self, type):
        return type in self._registry

    def __len__(self):
        return len(self._registry)
    
MODELS = Registry('MODELS')
DATASETS = Registry('DATASETS')
EVALUATIONS = Registry('EVALUATIONS')
TRANSFORMS = Registry('TRANSFORMS')

class Builder():
    def __init__(self, registry, **kwargs):
        self._registry = registry
        self.config = kwargs
    
    def build_module(self, type, config):
        if type not in self._registry:
            raise KeyError(f"'{type}' not registered in registry.")
        try:
            return self._registry.get(type)(**config)
        except Exception as e:
            raise RuntimeError(f"Error instantiating '{type}': {e}") from e     
