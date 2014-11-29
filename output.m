function y = output(x, w1, w2, w0)
	%just calculates the outputs
	v = x'*w1;
	y = (sigmoid(v-w0'))*w2;
