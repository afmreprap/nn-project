function y = voutput(x, w1, w2, w0)
	%calculates the outputs of each element of a vector of data points x
	y = ones(length(x),1);
	for i = 1:length(x),
		y(i) = output(x(:,i),w1, w2, w0);
	end