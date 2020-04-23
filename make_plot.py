import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

def plot_one(ax, data, title, xticklabels, yticklabels, cmap='Greens', annot=True, **kwargs):
    num_rows, num_cols = data.shape
    shifted_y_ticks = np.array([j - i for i,j in product(yticklabels,xticklabels)])
    shifted_y_ticks = np.unique(np.sort(shifted_y_ticks))
    shifted_data = np.zeros((len(shifted_y_ticks), len(xticklabels)))
    #for b, n in product(np.arange(num_rows), np.arange(num_cols)):
        #bn = np.argwhere(shifted_y_ticks == -(yticklabels[b] - xticklabels[n]))

        #print("b {} n {} b-n {}".format(b, n, bn[0]))
        #shifted_data[bn, n] = data[b,n]

    sns.heatmap(data,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=cmap,
        annot=annot,
        ax=ax,
        **kwargs)
    ax.title.set_text(title)

def plot_one_beta(ax, data, title, xticklabels, yticklabels, cmap='Greens', annot=True, **kwargs):
    num_rows, num_cols = data.shape
    shifted_y_ticks = np.array([j - i for i,j in product(yticklabels,xticklabels)])
    shifted_y_ticks = np.unique(np.sort(shifted_y_ticks))
    betas = np.zeros_like(data)
    for b, n in product(np.arange(num_rows), np.arange(num_cols)):
        betas[b, n] = yticklabels[b]#-xticklabels[n]

    sns.heatmap(betas,
        xticklabels=xticklabels,
        yticklabels=shifted_y_ticks,
        cmap=cmap,
        annot=annot,
        ax=ax,
        **kwargs)
    ax.title.set_text(title)


def plot_result(col, filename, xticklabels=None, yticklabels=None, annot=True):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        if type(data) == list:
            train_acc, test_acc, best_lr = data
        else:
            train_acc, test_acc, best_lr = data['train_acc'], data['test_acc'], data['best_lr']
            xticklabels, yticklabels = data['ns'], data['bs']

    plot_one(ax[0,col], train_acc, 'Train accuracy',
        xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)
    # maybe clim https://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib

    plot_one_beta(ax[1,col], train_acc, 'Train accuracy',
        xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)

    plot_one(ax[2,col], test_acc, 'Test accuracy',
        xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)

    #plot_one(ax[3,col], best_lr, 'Best lr',
     #   xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, cmap='Blues')

def ints(arr):
    return [int(x) if x % 1 == 0 else '' for x in arr]

# x = ns, y = bs
fig, ax = plt.subplots(3, 3)

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

#filename = 'results_last_layer.p'
filename = 'results.p'
filename ='results_b1e-3.5.p'
filename='results_random_labels.p'

with open(filename, "rb") as f:
    print(pickle.load(f))
plot_result(0, filename,
    xticklabels=np.arange(2, 4.51, 0.5).astype(int),
    yticklabels=np.arange(-2, 3.1, 1).astype(int))

plot_result(1, 'repro_real.p')
plot_result(2, 'repro_random.p')
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

# x axis labels
plt.setp(ax[-1, :], xlabel='$\log_{10}(N)$')
plt.setp(ax[:, 0], ylabel='$\log_{10}(N/\\beta)$')
fig = plt.gcf()
fig.set_size_inches(10, 15)
plt.savefig('train_acc.png')
plt.show()
