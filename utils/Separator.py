# -*- coding : utf-8 -*-
class Separator:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self._segments = None
        
    @property
    def segments(self):
        if self._segments is None:
            self._segments = []
        
            
        
        