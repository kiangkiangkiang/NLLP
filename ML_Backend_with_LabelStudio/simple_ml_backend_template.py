from label_studio_ml.model import LabelStudioMLBase

class SimpleMLBackend(LabelStudioMLBase):
    def __init__(self, **kwargs) -> None:
        super(SimpleMLBackend, self).__init__(**kwargs)
        pass


    def predict(self, tasks, **kwargs):
        pass


    def fit(self, annotation, workdir=None, **kwargs):
        pass
