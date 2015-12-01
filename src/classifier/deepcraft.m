%% Installing and compiling MatConvNet
% First, download the MatConvNet files from GitHub
% (https://github.com/vlfeat/matconvnet), and follow the online
% instructions for compiling (http://www.vlfeat.org/matconvnet/install/).
% 
% Before running any code for this assignment, you should make sure to run
% vl_setupnn so that matconvnet functions are added to your MATLAB path.
% 
% TODO: If you want to use the GPU + matconvnet for your final projects, 
% we will be distributing a modified version of matconvnet/matlab/vl_compilenn.m 
% that fixes this.
%

% Load imagenet. This is not on the repository and should be downloaded
% You can download* the pretrained network here: http://www.vlfeat.org/matconvnet/pretrained/
net = load('imagenet-vgg-verydeep-16.mat');

% To setup matconvnet follow the instructions below and add it to this
% current directory
run  matconvnet/matlab/vl_setupnn
addpath('matconvnet/matlab')
% vl_simplenn_display(net)

%%
% Now we need to preprocess the images in our dataset in order to make them
% compatible as inputs to our network. In the case of MatConvNet, this
% means resizing each image and subtracting the average pixel intensity
% from each pixel of each image. If you're unsure on how to do this, check
% the MatConvNet tutorial on how to use pretrained networks. Fill in the
% template function *load_and_preprocess_template.m* and perform the
% necessary operations on each image of each object.
%

% Here, you want to set up the data you're going to use, which is supposed
% ot be at '../../data/preprocessed'.
n_types = 2;
n_images = 61; % Hardcoded for now.
path_to_images = '../../data/processed/';
image_folder = {'forest/forest_1', 'snow/snow_1'};%, 'desert/desert_1','manmade/manmade_1'};
images = cell(n_types,1);
images_gray = cell(n_types,1);

for i=1:n_types
    path = strcat(path_to_images, image_folder{i},'/');
    images{i} = load_and_preprocess_n( path, net, n_images );
    images_gray{i} = load_and_preprocess_n_gray( path, net, n_images );
end

%%
% Get a feature vector for each image.

run matconvnet/matlab/vl_setupnn

features = cell(n_types,1);
for x=1:n_types
    n_imgs = numel(images{x});
    features_type = cell(n_imgs,1);
    for y=1:n_imgs
        features_type{y} = extractFeatures(images{x}{y},net, 36);
    end
    features{x} = features_type;
end
   
%%
% Get the gray feature vectors.
features_gray = cell(n_types,1);
for x=1:n_types
    n_imgs = numel(images{x});
    features_type_gray = cell(n_imgs,1);
    for y=1:n_imgs
        features_type_gray{y} = extractFeatures(images_gray{x}{y},net, 36);
    end
    features_gray{x} = features_type_gray;
end

%%
% SVM classifier. 
% Using 80% for training, 20% for test.

addpath('../../data/processed');

% Format features for SVM.
features_mat = [];
labels = [];
for i=1:n_types
    features_type = cell2mat( features{i} );
    features_mat = cat(1,features_mat,features_type);
    labels = cat(1, labels, i * ones( size(features_type,1), 1 ) );
end

acc = do_SVM(features_mat,labels);

% Accuracy should give ~100%

%% 

% Format gray features for SVM.
features_mat_gray = [];
labels_gray = [];
for i=1:n_types
    features_type_gray = cell2mat( features_gray{i} );
    features_mat_gray = cat(1,features_mat_gray,features_type_gray);
    labels_gray = cat(1, labels_gray, i * ones( size(features_type_gray,1), 1 ) );
end

acc_gray = do_SVM(features_mat_gray,labels_gray);

% Accuracy at around 98% as of now.