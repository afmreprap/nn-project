function y = backpro2(data, classification, hlayer)
	%the maximum number of iterations the algorithm runs, i is a counter for each iteration
	maxiter = 10^5;
	i = 0;
	%if the error is less than epsilon, the algorithm stops, otherwise it runs until maxiter
	epsilon = .01;
	err = inf;

	%randomly create weight vectors which should be named better
	lenData = length(data);
	n = length(data(:, 1));

	%w1 is the set of weight vectors going into the hidden layer of the form:
	%[a a1 a2...]
	%[b b1 b2...]
	%where [a;b] is one weight, and the length is that of the hidden layer
	w1 =  2*(rand(n, hlayer)-.5);

	%w0 is the bias of the hidden layer
	w0 = 2*10^-5*(rand(hlayer, 1)-.5);

	%and w2 is the weights from the hidden layer into the output, of the form [a;b;...]
	w2 = 2*(rand(hlayer, 1)-.5);
	inc =.1;

	%this stores the errors to be graphed later
	yd = zeros(maxiter, 1);

	%this repeatedly cycles through all of the data points, and for each point adjusts the weights by gradient descent
	while i < maxiter && err > epsilon
		%get the position of current data point
		pos = mod(i, lenData)+1;
		currData = data(:, pos);



		%calculate the output for the current classification
		ydelta =  -classification(pos) + output(currData, w1, w2, w0);

		%this term shows up a lot as one of the inside terms in the backpropogation chain-rule derivation
		v = (currData'*w1-w0')';

		%the w~delta terms are all part of the backpropogation in: w~-inc*ydelta()*w~delta
		w2delta = sigmoid(v);

		%this is tricky :(
		%repmat(A,n,m) expands a vector into a matrix of size n,m, 
		%so calling repmat on currData creates a matrix of hlayer repeated currData, if hlayer =3 and currData = [0;1], it would be:
		%the mess in the middle is part of the chain rule wrt to w1.  when deriviting the sigmoid, the sigmoid has form (1+exp(-v)).^-2).*exp(-v), this just does it for all of the points at once
		%when chain ruling the sigmoid, the inner term is just a bunch of w2*sum(x*w1), so all is left is x*w2, which is why you multipy by w2 and the currData
		w1delta = repmat((w2.*((1+exp(-v)).^-2).*exp(-v))',n,1).*repmat(currData, 1, hlayer);

		%the w0 term isn't multiplied by anything but -1, so there is a -1 instead
		w0delta = -(w2.*((1+exp(-v)).^-2).*exp(-v));

		%this is the gradient descent update step		
		w1 = w1 - ydelta*w1delta*inc;

		w2 = w2 - ydelta*w2delta*inc;

		w0 = w0 - ydelta*w0delta*inc;
		
		%calculates least squares error
		err = .5*(classification'-voutput(data, w1, w2, w0))'*(classification'-voutput(data, w1, w2, w0));
		yd(i+1) = err;

		i = i+1;
	end

	if err<epsilon && i<length(yd)
		yd = yd(1:i);
	end
	%graphs the error
	plot(yd)
	save output.txt -ascii w1 
	save output.txt -ascii -append w2
	save output.txt -ascii -append w0

	y = 0;
