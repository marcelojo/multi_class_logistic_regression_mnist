function image = get_image_mnist(data, index)
    image = uint8(reshape(data(:,index), 28, 28)*255);
endfunction