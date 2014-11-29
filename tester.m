function y = tester()
	data = [[0; 0] [1; 1] [0; 1] [1; 0]];
	clas = [0 0 1 1];

	y = backpro2(data, clas, 3);
	