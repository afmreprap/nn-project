function y = backpro(data, classification)
	maxiter = 10^5;
	i = 0;
	lenData = length(data);
	n = length(data(:, 1));
	w1 =  2*(rand(n, 1)-.5)
	w2 = 2*(rand(n+1, 1)-.5)
	w0 = 2*10^-5*(rand(n+1, 1)-.5)	
	inc =.2;

	yd = zeros(maxiter, 1);

	while i<maxiter,
		%get the current data point
		pos = mod(i, lenData)+1;
		currData = data(:, pos);

		%calculate the output for the current classification
		ydelta =  -classification(pos) + output(currData, w1, w2, w0);

		v = (w1'*currData-w0);

		w2delta = sigmoid(v);

		w1delta = (w2.*((1+exp(-v)).^-2).*exp(-v))'*ones(n+1,1)*currData;

		w0delta =  -(w2.*((1+exp(-v)).^-2).*exp(-v));

		w1 = w1 - ydelta*w1delta*inc;

		w2 = w2 - ydelta*w2delta*inc;

		w0 = w0 - ydelta*w0delta*inc;
		%a = input('next line');
		%plot(.5*(ydelta)^2);
		%hold on 
		yd(i+1) = .5*(classification'-voutput(data, w1, w2, w0))'*(classification'-voutput(data, w1, w2, w0));



		i = i+1;
	end

	plot(yd)

