function [vEnt] = calc_entropy(M)
% returns frame-wise entropy for input matrix M
EM = M.*log2(max(M,exp(-5)));
vEnt = -sum(EM,1);
