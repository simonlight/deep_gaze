'''
Created on 31 mai 2016

@author: wangxin
'''
class ImageBag(object):
    """Bag object"""
    def __init__ (self, name, label, features):
        self.name = name
        self.label = label
        self.features = features
    def __str__(self, *args, **kwargs):
        return "bag | name:%s, label:%s, instance number:%d, feature length:%d"\
            %(self.name, self.label, len(self.features), len(self.features[0]))
    
class GazeImageBag(ImageBag):
    def __init__(self, name,label,features,gazes):
        ImageBag.__init__(self, name, label, features)        
        self.gazes = gazes
    def __str__(self, *args, **kwargs):
        return "bag | name:%s, label:%s, instance number:%d, feature length:%d, gazes features:%s"\
            %(self.name, self.label, len(self.features), len(self.features[0]), str(self.gazes.items()))
        
