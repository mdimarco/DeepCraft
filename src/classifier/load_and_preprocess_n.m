function imgs = load_and_preprocess_n(path, net, n)
% Return cell array of preprocessed images from directory 'path'
    %net = load('imagenet-vgg-verydeep-16.mat');

    files = dir(path);
    files = files(3:end); % truncate '.' and '..'
    n_files = length(files);
    if n_files < n 
        disp(sprintf('%s: There are only %d files in this folder (you wanted %d).', path, n_files, n));
        return;
    end
    
    imgs = cell(n,1);

    for i=1:n
        im = imread(strcat(path,files(i).name)) ;
        im_ = single(im) ; % note: 0-255 range
        im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
        im_ = im_ - net.normalization.averageImage ;
        imgs{i} = im_;
    end
    
    
end