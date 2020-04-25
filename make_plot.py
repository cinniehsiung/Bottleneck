import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

col_titles = ['Random labels', 'Real labels']

def plot_one(row, col, data, xticklabels, yticklabels, shift=True, cmap='RdBu_r', annot=True, **kwargs):
    ns, bs = xticklabels, yticklabels

    num_rows, num_cols = data.shape
    shifted_y_ticks = np.array([j - i for i,j in product(yticklabels,xticklabels)])
    shifted_y_ticks = np.unique(np.sort(shifted_y_ticks))
    shifted_data = np.full((len(shifted_y_ticks), len(xticklabels)), np.nan)

    for b, n in product(np.arange(num_rows), np.arange(num_cols)):
        bn = np.argwhere(shifted_y_ticks == -(yticklabels[b] - xticklabels[n]))
        shifted_data[bn, n] = data[b,n]

    # limit range
    start, = np.where(shifted_y_ticks == 1.5)[0]
    stop, = np.where(shifted_y_ticks == 5.5)[0]
    shifted_y_ticks = shifted_y_ticks[start:stop+1]
    shifted_data = shifted_data[start:stop+1,:]

    # cmap = plt.get_cmap('RdBu_r')
    ax[row, col].set_aspect(abs(ns[0]-ns[-1])/abs(shifted_y_ticks[0] - shifted_y_ticks[-1]))
    heatmap = ax[row, col].pcolormesh(ns, shifted_y_ticks ,shifted_data if shift else data, 
        # xticklabels=ns,
        # yticklabels=shifted_y_ticks if shift else bs,
        cmap=cmap,
        # annot=annot,
        # label='test',
        **kwargs)
    # ax[row, col].set_xticks(ns)
    # ax[row, col].set_yticks(shifted_y_ticks)

    cbar = plt.colorbar(heatmap, ax=ax[row, col], ticks=[0,1], fraction=0.046, pad=0.04)
    cbar.set_label(('Training' if row == 0 else 'Testing') + ' Accuracy', rotation=270)
    cbar.ax.set_yticklabels([0,1])
    
    if row == 0:
        ax[row, col].title.set_text(col_titles[col])


def plot_result(col, filename, xticklabels=None, yticklabels=None, annot=True):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        if type(data) == list:
            train_acc, test_acc, best_lr = data
        else:
            train_acc, test_acc, best_lr = data['train_acc'], data['test_acc'], data['best_lr']
            xticklabels, yticklabels = data['ns'], data['bs']
        train_acc[train_acc==-1] = np.nan
        test_acc[test_acc==-1] = np.nan

    plot_one(0, col, train_acc,
        xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)
    # maybe clim https://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib

    plot_one(1, col, test_acc,
        xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)

    # plot_one(ax[1,col], test_acc, 'Test accuracy',
        # xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)

    #plot_one(ax[3,col], best_lr, 'Best lr',
     #   xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, cmap='Blues')

def ints(arr):
    return [int(x) if x % 1 == 0 else '' for x in arr]

ax = None
def make_plot(files):
    global ax
    fig, ax = plt.subplots(2, max(2, len(files)))
    for i, file in enumerate(files):
        plot_result(i, file, annot=False)
    # x axis labels
    plt.setp(ax[-1, :], xlabel='$\log_{10}(N)$')
    plt.setp(ax[:, 0], ylabel='$\log_{10}(N/\\beta)$')
    fig = plt.gcf()
    fig.set_size_inches(7, 6)
    plt.savefig('transition.pdf')
    plt.show()
make_plot(['final_rand.p', 'final_norand.p'])

# x = ns, y = bs
# make_plot(['scaled_alpha_eps_4_rand.p', 'eps0.0001_rand.p', 'repro_random.p', 'eps0.01_rand.p'])

# plt.suptitle('repro (real, random), eps (real, random)')
# plot_result(0, 'repro_real.p')
# plot_result(1, 'repro_random.p')
# plot_result(2, 'eps_norand.p')
# plot_result(3, 'eps_rand.p')
# filename='results_random_labels.p'

# still can't reproduce previous examples
# plt.suptitle('stop when test accuracy stops increasing (norand/rand)')
# plot_result(0, '1920alpha_ES_norand.p')
# plot_result(1, '1920alpha_ES_rand.p')

# last layer random; sharp diagonal but not sure about what the y-axis was
# plot_result(0, 'results_random_labels.p',
#     xticklabels=np.arange(2, 4.51, 1).astype(int),
#     yticklabels=np.arange(-2, 3.1, 1).astype(int))

# these results have all 100% train accuracy
# plot_result(0, 'no_mul_rand.p')
# plot_result(1, 'no_mul_norand.p')
# plot_result(2, '1920alpha_rand.p')

# plot_result(2, 'mul1920_rand.p') # all 1's
# plot_result(3, 'mul1920_norand.p')

# last layer nonrandom, pretty uniform
# plot_result(1, 'results_last_layer.p',
#     xticklabels=np.arange(2, 4.51, 1).astype(int),
#     yticklabels=np.arange(-2, 3.1, 1).astype(int))

# plot_result(2, 'all_layers.p', # also pretty uniform
#     xticklabels=np.arange(2, 4.51, 1).astype(int),
#     yticklabels=np.arange(-2, 3.1, 1).astype(int))

# initial partial results
# plot_result(1, 'results.p',
#   xticklabels=ints(np.arange(2, 4.51, 0.25)),
#   yticklabels=ints(np.arange(-3.5, 3, 0.25)), annot=False)

# plot_result(2, 'results_b1e-3.5.p',
#   xticklabels=ints(np.arange(2, 4.51, 0.25)),
#   yticklabels=ints(np.arange(-3, 3, 0.25)), annot=False)

# don't want larger max_alpha--too blurry
# plot_result(1, 'alpha1random.p',
#     xticklabels=np.arange(2, 4.51, 1).astype(int),
#     yticklabels=np.arange(-2, 3.1, 1).astype(int))
# plot_result(2, 'alpha2.5random.p',
#     xticklabels=np.arange(2, 4.51, 1).astype(int),
#     yticklabels=np.arange(-2, 3.1, 1).astype(int))

# here we didn't normalize the KL loss term by max_alpha
# this was just weird, so we don't like it
# plot_result(3, 'alpha0.7random_nonorm.p',
#     xticklabels=np.arange(2, 4.51, 1).astype(int),
#     yticklabels=np.arange(-2, 3.1, 1).astype(int))

# plot_result(1, 'all_layers_random.p', # 0.7 slow, only a few done
#     xticklabels=np.arange(2, 4.51, 1).astype(int),
#     yticklabels=np.arange(-2, 3.1, 1).astype(int))


