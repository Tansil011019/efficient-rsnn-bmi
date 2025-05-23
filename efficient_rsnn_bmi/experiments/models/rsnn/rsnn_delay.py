from stork.models import RecurrentSpikingModel

class DelayRecurrentSpikingModel(RecurrentSpikingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)