clear
clf
clc

% Load training data
images_train = loadMNISTImages('./mnist/train-images.idx3-ubyte');
images_train = images_train';
labels_train = loadMNISTLabels('./mnist/train-labels.idx1-ubyte');

% Just to be faster, we can train only in a part of the entire trainind dataset
train_qty = 1000;   % Max of 60k


data_train = images_train(1:train_qty, :);
label_train = labels_train(1:train_qty);


% Training
a_param = train_regression(data_train, label_train);


% Testing
% Load data
images_test = loadMNISTImages('./mnist/t10k-images.idx3-ubyte');
images_test = images_test';
labels_test = loadMNISTLabels('./mnist/t10k-labels.idx1-ubyte');

% Classify
right = 0;
wrong = 0;
for i = 1:size(images_test)(1)
    [prob, pred] = test_regression(a_param, images_test(i,:));
    
    if(pred == labels_test(i))
        right = right + 1;
    else 
        wrong = wrong + 1;
    end
endfor

accuracy = right / (right + wrong);
printf("Hits: %d, Miss: %d. Total: %d\n", right, wrong, right + wrong);
printf("Multi-class Logistic Regression accuracy: %1.2f%%\n\n", accuracy * 100);

% Interactive testing


for i = 1:1000
    A = uint8(reshape(images_test(i,:), 28, 28) * 255);
    imshow(A);
    
    [probability, prediction] = test_regression(a_param, images_test(i,:));
    msg = sprintf(" Number %d predicted with %1.2f%% of certainty\n\n", (prediction), (probability * 100));
    title(msg);
    
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
endfor

