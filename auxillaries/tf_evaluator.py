class Tf_evaluator:
    def __init__(self):
        self._session = None
        self._feed_dict = None

    def initialize(self, session, feed_dict):
        if (self._session != None) | (self._feed_dict != None):
            raise Exception('[Tf_logger]. Already initialized')

        self._session = session
        self._feed_dict = feed_dict

    def evaluate(self, tensor, session=None, feed_dict=None, initializer=None):
        if session == None:
            session = self._session
        if feed_dict == None:
            feed_dict = self._feed_dict

        if (session == None) | (feed_dict == None):
            raise Exception('[Tf_logger]. Not initialized')

        if initializer != None:
            session.run(initializer)

        res = session.run(tensor, feed_dict)
        return res
