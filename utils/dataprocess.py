import imageio
def read_from_folder(folder_path, img_type, img_num):
    """
    Get the image (.png) file from a given folder and return an array which contain all
    images from the folder
    :param folder_path: A string where the function read image from
    :param img_type: A string that the type of image .jpg or .png ...
    :param img_num: An int value how many images will be read from folder_path
    :return X_array: A python array contain all image from folder_path
    """
    X_array = []
    for i in range(1,img_num):
        image_name = ''.join([str(i), img_type])
        image_full_name = ''.join([folder_path, image_name])
        image = imageio.imread(image_full_name)
        X_array.append(image)
    return X_array


def get_image_box(index, h5pyFile):
    """
    Get the box information from digitStruct.mat file
    .mat file contain the box location for each digits in an image
    :param index: int number means which image that read from digitStruct.mat
    :param h5pyFile: A h5py file which get from digitStruct.mat
    :return box_info: A python dictionary which is the box information of the index image
    """
    box_info = {}
    box_info['height'] = []
    box_info['left'] = []
    box_info['top'] = []
    box_info['width'] = []
    box_info['label'] = []
    bbox =  h5pyFile['/digitStruct/bbox'][index]
    def print_para(ind, obj):
        value = []
        if obj.shape[0] == 1:
            value.append(obj[0][0])
        else:
            for i in range(obj.shape[0]):
                value.append(int(h5pyFile[obj[i][0]][0][0]))
        box_info[ind] = value
    h5pyFile[bbox[0]].visititems(print_para)
    return box_info


def image_crop(orig_image, box_info):
    """
    crop the image base on the box information, the cropped image should contain all the digits
    in that image and And expand the border by 30 percent
    :param orig_image: A np.array which is the image before crop
    :param box_info: A python dictionary which is the box information of the index image
    :return: image_crop: A np.array which is the image after crop
             image_crop_range: A python array which contains box size information
    """
    orig_image_shape = orig_image.shape
    orig_Height = box_info['height']
    orig_Left = box_info['left']
    orig_Top = box_info['top']
    orig_Width = box_info['width']

    bounding_box_top = int(min(orig_Top))
    bounding_box_left = int(min(orig_Left))
    bounding_box_right = int(orig_Left[-1] + orig_Width[-1])
    bounding_box_below = int(max((orig_Top[i]+orig_Height[i])for i in range(len(orig_Height))))
    L = int(bounding_box_right - bounding_box_left)
    H = int(bounding_box_below - bounding_box_top)
    L_pad = int(round(L*0.15))
    H_pad = int(round(H * 0.15))

    crop_top = int(bounding_box_top - H_pad)
    if crop_top < 0:
        crop_top = 0
    crop_below = int(bounding_box_below + H_pad)
    if crop_below > orig_image_shape[0]:
        crop_below = int(orig_image_shape[0])
    crop_left = int(bounding_box_left - L_pad)
    if crop_left < 0:
        crop_left = 0
    crop_right = int(bounding_box_right + L_pad)
    if crop_right > orig_image_shape[1]:
        crop_right = int(orig_image_shape[1])
    image_crop = orig_image[crop_top: crop_below, crop_left: crop_right]
    image_crop_range = [crop_top, crop_below, crop_left, crop_right]
    #print(crop_top, crop_below, crop_left, crop_right)
    return image_crop, image_crop_range

















