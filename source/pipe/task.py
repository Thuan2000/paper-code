class Task(object):
    Frame = 'TaskHanleFrameEventIfNoFace'
    Face = 'TaskHandleFace'  # TODO: Maybe rename to object when
    Event = 'TaskHandleEvent'

    def __init__(self, type="Unknown"):
        self.type = type
        self.packaging = False
        self.data = None

    def package(self, **args):
        assert not self.packaging
        self.packaging = True
        self.data = args

    def depackage(self):
        assert self.packaging
        return self.data
