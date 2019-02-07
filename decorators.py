def partially_applied(func):
    def outer_wrapper(**kwargs):
        def inner_wrapper(*args):
            return func(*args, **kwargs)
        return inner_wrapper
    return outer_wrapper