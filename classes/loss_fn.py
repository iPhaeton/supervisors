class Triplet_loss_fn:
    def __init__(self, loss_fn, metric, margin):
        self.loss_fn = loss_fn
        self.metric = metric
        self.margin = margin

    def __call__(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs, metric=self.metric, margin=self.margin)