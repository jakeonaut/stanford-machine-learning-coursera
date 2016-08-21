function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

	pred = (pval < epsilon);
	%the number of 1s will be equivalent to the sum
	%true positives, we predict anomaly (pval < epsilon) and they match
	tp_mat = (pred == 1) & (yval == 1);
	tp = sum(tp_mat); 
	%false positives, we predict anomal (pval < epsilon), but they don't match
	fp_mat = (pred == 1) & (yval == 0);
	fp = sum(fp_mat);
	%false negatives, we do not predict anomal (pred == 0), and they don't match
	fn_mat = (pred == 0) & (yval == 1);
	fn = sum(fn_mat);
	
	prec = tp / (tp + fp);
	rec = tp / (tp + fn);
	F1 = (2*prec*rec)/(prec+rec);











    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
