import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn.functional as F

POLY_DEGREE = 5
batch_size = 32
lrs = 0.6
batch_idx_max = 2000
dclrbatch = 300

W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5

def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x @ W_target + b_target.item()

def get_batch():
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y

fc = torch.nn.Linear(W_target.size(0), 1)

def data_gen():
    lr = lrs
    batch_idx = 0
    while 1:
        if batch_idx < batch_idx_max:
            batch_idx +=1
            batch_x, batch_y = get_batch()
            fc.zero_grad()
            output = F.smooth_l1_loss(fc(batch_x), batch_y)
            loss = output.item()
            output.backward()
            
            if batch_idx % dclrbatch == 0:
                print(f'dec:{batch_idx//dclrbatch},lr:{lr}')
                log(loss,batch_idx)
                lr -= 0.00001
            
            # Apply gradients
            for param in fc.parameters():
                param.data.add_(-lr * param.grad.data)
        yield batch_idx,np.array(loss)

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.3f} x^{} '.format(w, i + 1)
    result += '{:+.3f}'.format(b[0])
    return result

def log(loss=0,batch_idx=0):
    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
    print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        #ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
                              repeat=False, init_func=init)
plt.show()
print('=====final====')
log()