
class size(object):

    def __init__(self, size, format = 'hw'):

        if format == 'hw':

            self.height = size[0]

            self.width = size[1]

        elif format == 'wh':

            self.height = size[1]

            self.width = size[0]

        else:

            ValueError('Incorrect Error')

        self.format = format
