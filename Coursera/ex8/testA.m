
A=[1,2,3,4;5,6,7,8;9,10,11,12;13,14,15,16;17,18,19,20];
A = A + 5;
A
stepsize = (max(A) - min(A)) / 3;
for epsilon = min(A):stepsize:max(A)
	epsilon
	A(epsilon:epsilon + stepsize)
end
