#
# The codes are used for implementing CTPN for scene text detection, described in: 
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
# 
# These demo codes (with our trained model) are for text-line detection (without 
# side-refiement part).  
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
# 
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#

from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import cv2, os, sys
sys.path.append('../../CaffeCTPN')
import caffe
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer

DEMO_IMAGE_DIR="E:/container_ocr/CTPN/demo_images"


#sys.path.append('../caffe-master/python')


caffe.set_mode_cpu()
global text_proposals_detector
global text_detector
global caffe_model
def init(model_file,trained_file):
# initialize the detectors
    global text_proposals_detector
    global text_detector
    caffe_model=CaffeModel(model_file, trained_file)
    text_proposals_detector=TextProposalDetector(caffe_model)
    text_detector=TextDetector(text_proposals_detector)



'''
for im_name in demo_imnames:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%im_name

    im_file=osp.join(DEMO_IMAGE_DIR, im_name)
    im=cv2.imread(im_file)

    timer.tic()

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)

    print "Number of the detected text lines: %s"%len(text_lines)
    print "Time: %f"%timer.toc()

    im_with_text_lines=draw_boxes(im, text_lines, is_display=False, caption=im_name, wait=False)
    cv2.imwrite("%s_res.jpg" % im_name[:-4], im_with_text_lines)

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "Thank you for trying our demo. Press any key to exit..."
cv2.waitKey(0)
'''



def detect(im_path):
    global text_detector
    im=cv2.imread(im_path)
    text_lines=text_detector.detect(im)
    '''
    for i in range(text_lines.shape[0]):
        for j in range(text_lines.shape[1]):
            print(text_lines[i][j])
            print(" ")
    '''
    return [text_lines,]

def detect1(img_array,shapeinfo):
    im=img_array.reshape(shapeinfo[0],shapeinfo[1],shapeinfo[2])
    text_lines=text_detector.detect(img_array)
    for i in range(text_lines.shape[0]):
        for j in range(text_lines.shape[1]):
            print(text_lines[i][j])
            print(" ")
    return [text_lines,]

#init(NET_DEF_FILE,MODEL_FILE)

from socket import *
from time import ctime
import json
def ctpn_server():

    global text_detector
    HOST=''
    PORT=6002
    BUFSIZ=1024
    ADDR=(HOST, PORT)
    sock=socket(AF_INET, SOCK_STREAM)
    sock.bind(ADDR)
    sock.listen(5)
    while True:
        print('waiting for connection')
        tcpClientSock, addr=sock.accept()
        print('connect from ', addr)
        #while True:
        try:
            data=tcpClientSock.recv(BUFSIZ)
        except:
            print(e)
            tcpClientSock.close()
            break
        if not data:
            print('not data')
            err_response={"error_code":1}
            response_str=json.dumps(err_response)+'\0'
            tcpClientSock.send(response_str.encode('utf8'))
            break
        imgpath=data.decode('utf8')
        img=cv2.imread(imgpath,1)
        text_lines=text_detector.detect(img)
        response={}
        for i in range(text_lines.shape[0]):
            response[i]=text_lines[i].tolist()
        response["error_code"]=0
        response["boxnum"]=text_lines.shape[0]
        response_str=json.dumps(response)+'\0'
        tcpClientSock.send(response_str.encode('utf8'))
        print([ctime()], ':', imgpath)
        tcpClientSock.close()
    sock.close()
if __name__ == '__main__':
    global NET_DEF_FILE
    global MODEL_FILE
    NET_DEF_FILE="./models/deploy.prototxt"
    MODEL_FILE="./models/ctpn_trained_model.caffemodel"
    init(NET_DEF_FILE,MODEL_FILE)
    ctpn_server()
