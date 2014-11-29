function y = sigmoid(x)
	y = 1./(1+exp(-x)); %by taking the ./ and the exp of x instead of -x, can take sigmoid by value of arrays
	