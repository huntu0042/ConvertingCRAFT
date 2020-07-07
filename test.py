import file_utils
import os
import tensorflow as tf
import time
import cv2
import numpy as np

import imgproc
import craft_utils
from craft_net import CRAFT

result_folder = './synth_result/'

""" For test images in a folder """
image_list_ic15, _, _ = file_utils.get_files('./eval_data_ic15/')
image_list_ours, _, _ = file_utils.get_files('./choice/')

if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

canvas_size = int(2240)
mag_ratio = float(2)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly,filename,result_folder=result_folder):
    t0 = time.time()
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    #cv2.imwrite("test.jpg",x)
    print("###")
    x = tf.expand_dims(x,0)
    print(x.shape)

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0,:,:,0].numpy()
    score_link = y[0,:,:,1].numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    #print("score")
    #print(ret_score_text.shape)
    cv2.imwrite(result_folder + filename + "_mask.jpg",ret_score_text)


    #if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def test(pre_model,res_dir = result_folder,mode=0): ## mode 0 = ic15 1 = ours
    # load net
    net = CRAFT()     # initialize

    text_threshold = float(0.7)
    low_text = float(0.4)
    link_threshold = float(0.4)
    cuda = True
    poly = False

    print('Loading weights from checkpoint {}'.format(pre_model))
    #loaded_model = tf.keras.models.load_model(pre_model)
    loaded_model = net.load_weights(pre_model).expect_partial()
    print(loaded_model)

    t = time.time()
    print("#############")
    print(net)



    if mode != 0:
        image_list = image_list_ours
    else:
        image_list = image_list_ic15

    print(image_list)


    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)


        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        save_file_name = filename

        bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, filename)
        # save score text

        mask_file = res_dir + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=res_dir)

    print("Eval elapsed time : {}s".format(time.time() - t))

if __name__ == '__main__':
    test("checkpoints/my_checkpoint","test_result/",0)