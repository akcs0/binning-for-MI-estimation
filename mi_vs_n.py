import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import math 
from scipy.interpolate import make_interp_spline, BSpline

# decide on setup
num_Y_labels = 3
num_diff_samples = 10 # dots on each horizontal line
num_diff_bins = 5 # horizontal lines, one for each bin

def graph_MI_vs_samples (num_diff_samples, num_diff_bins):
	# set up N values
	N_values = []
	for i in range(num_diff_samples):
		N_values.append(500 * (i + 1)) # increase sample num by 500 each time
	# set up b values
	b_values = []
	for i in range(num_diff_bins):
		b_values.append(50 * (i + 1)) # increase bin num by 50 each time
	# set up variables to graph, calculate points
	x = N_values
	ys = [[0 for i in range(len(N_values))] for j in range(len(b_values))]
	c = 0
	for i in range(len(b_values)):
		b = b_values[i]
		for j in range(len(N_values)):
			N = N_values[j]
			ys[i][j] = calculate_MI(num_Y_labels, N, b)
	# plot setup
	plt.clf()
	plt.title("MI vs. N for different numbers of bins")
	plt.xlabel('number of samples')
	plt.ylabel('mutual information')
	xnew = np.linspace(0, N_values[-1], 300) # for smooth line
	# plot scatterplot and smooth lines
	for i in range(len(b_values)):
		# preparing smooth line
		spl = make_interp_spline(x, ys[i], k=3)
		ys_smooth = spl(xnew)
		# plot points and line
		scatter = plt.plot(x, ys[i], 'o')
		line = plt.plot(xnew, ys_smooth, color = scatter[0].get_color(), label = b_values[i])
	plt.legend(loc = 'upper right')
	plt.show()

def calculate_MI (num_Y_labels, N, b):
	# generate Y samples (binomial distribution)
	Y = []
	count = 0 
	for i in range(num_Y_labels):
		Y.append(count)
		count = count + 1
	Y_samples = np.random.choice(Y, size=N) # all labels equal prob for now
	# generate corresponding values of X (normal distribution for each label)
	X = []
	for y in Y_samples:
		for i in range(num_Y_labels):
			if y == Y[i]:
				# normal distribution of X for each label in Y
				X.append(float(np.random.normal(2 * i,1,1)))
				break;
	# binning
	hist = count, bins, ignored = plt.hist(X, b) # only binning X so far
	# calculating bin chart
	chart = [[0 for i in range(b)] for j in range(num_Y_labels)] 
	for x in X:
		for i in range(b):
			if x >= bins[i] and x < bins[i+1]:
				chart[Y_samples[X.index(x)]][i] = chart[Y_samples[X.index(x)]][i]+1
				break;
			if x == bins[i+1]: # last bin is closed
				chart[Y_samples[X.index(x)]][i] = chart[Y_samples[X.index(x)]][i]+1
				break;
	# preparing to calculate experimental p(y)
	count_y = [0 for i in range(num_Y_labels)]
	for y in Y_samples:
		for i in range(len(Y)):
			if y == Y[i]:
				count_y[i] = count_y[i] + 1
				break;
	# calculating MI
	MI = 0; p_x = 0; p_y = 0; p_xy = 0; e = math.e ** -10
	for y in Y:
		# calculating p(y)
		for i in range(num_Y_labels):
			if y == Y[i]:
				p_y = count_y[i]/N
				break;
		for x in range(b):
			# calculating p(x): probability of being in this bin
			p_x = count[x]/N
			# calculating p(x,y): total probability not exactly 1
			p_xy = chart[y][x] / N # num in our bin / normalized denominator
			# add to MI sum if 
			if min(p_x, p_y, p_xy) >= e:
				MI = MI + (p_xy * math.log((p_xy/(p_x * p_y)), 2))
	# return MI
	return MI

# results
graph_MI_vs_samples(num_diff_samples, num_diff_bins)
