



class backboneinterface(object):
    def __init__(self, pretrained:bool):
        super(backboneinterface).__init__()
        self.pretrained = pretrained

    def define_backbone(self, type):
        if(type == 'vgg16'):
            from backbone.Vgg16 import vgg16
            return vgg16
        elif(type == 'resnet101'):
            raise ValueError
    def feature(self):
        raise NotImplementedError

