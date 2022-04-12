raw_dir = 'Camelyon16\Data\Tumor\Thumbnails'
new_dir = 'Camelyon16\Data\Tumor\Tissue-BG Masks'
patient_list = {dir(raw_dir).name}
patient_list = patient_list(3:end)

for i = 1:length(patient_list)
    slide_list_dir = strcat(raw_dir,'\',string(patient_list(i)));
    I = imread(slide_list_dir);

    level = graythresh(I);
    BW = imbinarize(I,level);
    BW = im2uint8(BW);
    BW = rgb2gray(BW);
    BW1 = im2bw(BW);
    BW1 = ~bwareaopen(~BW1, 1500);
    BW2 = bwareaopen(BW1, 1500);
    BW2 = ~BW2;
    BW2 = im2bw(BW2,0.5);
    
    se = strel('disk',5);
    BW2 = imdilate(BW2,se);

    img_size = size(BW2)
    height = img_size(1);
    width = img_size(2);

    BW2(1:round(height*0.1),:) = 0;
    BW2(round(height*0.9):end,:) = 0;
    BW2(:,1:round(width*0.1)) = 0;
    BW2(:,round(width*0.9):end) = 0;

    BW2 = im2bw(BW2,0.5);

    save_dir = strcat(new_dir,'\',string(patient_list(i)))
    imwrite(BW2,save_dir)
end
