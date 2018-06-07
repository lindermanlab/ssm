import autograd.numpy as np
import autograd.numpy.random as npr

def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def ensure_args_are_lists(f):
    def wrapper(self, datas, inputs=None, masks=None, **kwargs):
        datas = [datas] if not isinstance(datas, list) else datas
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], self.M)) for data in datas]
        elif not isinstance(inputs, list):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, list):
            masks = [masks]

        return f(self, datas=datas, inputs=inputs, masks=masks, **kwargs)

    return wrapper


def ensure_elbo_args_are_lists(f):
    def wrapper(self, variational_params, datas, inputs=None, masks=None, **kwargs):
        datas = [datas] if not isinstance(datas, list) else datas
        
        if inputs is None:
            inputs = [np.zeros((data.shape[0], self.M)) for data in datas]
        elif not isinstance(inputs, list):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, list):
            masks = [masks]

        return f(self, variational_params, datas, inputs=inputs, masks=masks, **kwargs)

    return wrapper


def ensure_args_not_none(f):
    def wrapper(self, data, input=None, mask=None, **kwargs):
        assert data is not None
        input = np.zeros((data.shape[0], self.M)) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, data=data, input=input, mask=mask, **kwargs)
    return wrapper
