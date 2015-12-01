function features = getFeatures(net, image, layer)

%net = load('imagenet-vgg-verydeep-16.mat');

% This function serves to output a feature vector for a given layer. You
% will want to review the MatConvNet tutorial on using pre-trained networks
% to complete this function. Hint: you will need to use the function
% *vl_simplenn.m* to run the image through the network. You then need to
% choose which layer you would like to access. You may choose to reshape
% your output within this function or in your module's code.
% ===========TYPE YOUR CODE HERE================

res = vl_simplenn(net, image) ;
n = prod(size(res(layer).x));
features = reshape(res(layer).x, [1,n]);

end