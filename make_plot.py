import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_one(ax, data, title, xticklabels, yticklabels, cmap='Greens', annot=True, **kwargs):
	sns.heatmap(data,
		xticklabels=xticklabels,
		yticklabels=yticklabels,
		cmap=cmap,
		annot=annot,
		ax=ax,
		**kwargs)
	ax.title.set_text(title)

def plot_result(col, filename, xticklabels, yticklabels, annot=True):
	with open(filename, 'rb') as f:
		train_acc, test_acc, best_lr = pickle.load(f)

	plot_one(ax[0,col], train_acc, 'Train accuracy',
		xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)
	# maybe clim https://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib

	plot_one(ax[1,col], test_acc, 'Test accuracy',
		xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, vmin=0, vmax=1)

	plot_one(ax[2,col], best_lr, 'Best lr',
		xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, cmap='Blues')

def ints(arr):
	return [int(x) if x % 1 == 0 else '' for x in arr]

# x = ns, y = bs
fig, ax = plt.subplots(3, 2)

plot_result(0, 'results_conv_layers.p',
	xticklabels=np.arange(2, 4.51, 1).astype(int),
	yticklabels=np.arange(-2, 3.1, 1).astype(int))

plot_result(1, 'results_last_layer.p',
	xticklabels=np.arange(2, 4.51, 1).astype(int),
	yticklabels=np.arange(-2, 3.1, 1).astype(int))

# initial partial results
# plot_result(1, 'results.p',
# 	xticklabels=ints(np.arange(2, 4.51, 0.25)),
# 	yticklabels=ints(np.arange(-3.5, 3, 0.25)), annot=False)

# plot_result(2, 'results_b1e-3.5.p',
# 	xticklabels=ints(np.arange(2, 4.51, 0.25)),
# 	yticklabels=ints(np.arange(-3, 3, 0.25)), annot=False)

# x axis labels
plt.setp(ax[-1, :], xlabel='N')
plt.setp(ax[:, 0], ylabel='beta')
plt.tight_layout()
plt.savefig('train_acc.png')
plt.show()
