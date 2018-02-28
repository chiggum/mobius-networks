from keras.regularizers import Regularizer
from keras import backend as K

class OrthogonalRegularizer(Regularizer):
    def __init__(self, eye_len, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.eye = K.eye(eye_len)

    def __call__(self, y):
        regularization = 0.
        x = K.dot(K.transpose(y), y) - self.eye
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}