class EventAggregator:
    def __init__(self):
        self._listeners = {}

    def emit(self, name, *args, **kwargs):
        if name in self._listeners:
            for listener in self._listeners[name]:
                listener(*args, **kwargs)
    
    def add_listener(self, name, listener):
        if callable(listener) == False:
            raise Exception(f'[EventAggregator] Listener must be callable. Got {type(listener)} instead')

        if name in self._listeners:
            self._listeners[name].append(listener)
        else:
            self._listeners[name] = [listener]