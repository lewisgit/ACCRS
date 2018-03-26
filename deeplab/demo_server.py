import numpy as np
import cv2
import os
import sys

sys.path.append('../../CaffeDeeplab')
import caffe
import basicprocess
import postprocess

tmpPath = os.path.join('.', 'tmp')

#caffe.set_device(0)
caffe.set_mode_cpu()
prototxt = os.path.join('.', 'config', 'test.prototxt')
model = os.path.join('.', 'model', 'train_iter_1000.caffemodel')
net = caffe.Net(prototxt, model, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
print(net.blobs['data'].data.shape)

transformer.set_mean('data',np.array([104.008,116.669,122.675]))
#transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2,1,0))
transformer.set_transpose('data', (2,0,1))
#transformer.set_channel_swap('data', (2,1,0))


def detectContainer(imgPath):
    global net
    global transformer
    # resize src image and generate .ppm
    orgImg = cv2.imread(imgPath)
    newImg = basicprocess.imgPretreat(orgImg)
    imgName = os.path.split(imgPath)[1]
    imgNameNoPath = os.path.splitext(imgName)[0]
    print(imgName, imgNameNoPath)
    cv2.imwrite(os.path.join(tmpPath, 'resizedImg', imgName), newImg)
    cv2.imwrite(os.path.join(tmpPath, 'ppmImg', imgNameNoPath+'.ppm'), newImg)
    ppmImg = open(os.path.join(tmpPath, 'ppmImg', imgNameNoPath+'.ppm'), 'r+')
    ppmImg.writelines("P6\n")
    ppmImg.writelines("                          ")
    ppmImg.close()
    
    #write .txt
    testTxt = open(os.path.join(tmpPath, 'test.txt'), 'w')
    testTxt.writelines('\\' + imgName + '\n')
    testTxt.close()
    
    testIdTxt = open(os.path.join(tmpPath, 'test_id.txt'), 'w')
    testIdTxt.writelines(imgNameNoPath + '\n')
    testIdTxt.close()
    
    # generate the net and it will generate .mat in ./tmp/matresult
    
    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    # prototxt = os.path.join('.', 'config', 'test.prototxt')
    # model = os.path.join('.', 'model', 'train_iter_1000.caffemodel')
    # net = caffe.Net(prototxt, model, caffe.TEST)

    #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})


    #transformer.set_mean('data',np.array([104.008,116.669,122.675]))
    #transformer.set_raw_scale('data', 255)
    #transformer.set_channel_swap('data', (2,1,0))
    #transformer.set_transpose('data', (2,0,1))
    meanvalue=[104.008,116.669,122.675]
    orgimg=cv2.imread(os.path.join(tmpPath, 'resizedImg', imgName))
    padheight=513-orgimg.shape[0]
    padwidth=513-orgimg.shape[1]
    borderimg=cv2.copyMakeBorder(orgimg,0,padheight,0,padwidth,cv2.BORDER_CONSTANT,value=meanvalue)

    img = np.array(borderimg)
    #img = caffe.io.load_image(os.path.join(tmpPath, 'resizedImg', imgName))
    print(img.shape)
    newnewimg=img.astype(np.double)
    newnew=transformer.preprocess('data', img)
    print(newnew.shape)
    net.blobs['data'].data[...] = newnew
    net.forward()
    

    denseCrfCmd = '.\\densecrf\\densecrf.exe -id .\\tmp\\ppmImg -fd .\\tmp\\matresult -sd .\\tmp\\densecrf -i 10 -px 3 -py 3 -pw 3 -bx 49 -by 49 -br 5 -bg 5 -bb 5 -bw 4'
    os.system(denseCrfCmd)
    
    # resize to orginal size
    labelGray = basicprocess.bin2png(os.path.join(tmpPath, 'densecrf',imgNameNoPath+'.bin'))
    orgImgSize = np.shape(orgImg)[0:2][::-1]
    labelGray = cv2.resize(labelGray, orgImgSize)
    maskImg = basicprocess.mask(orgImg, labelGray)
    
    #cv2.imwrite(imgPath+'_label.jpg', maskImg)
    
    processedImg, imgArea = postprocess.regionNMS(labelGray)    
    processedImg, topR, bottomR, leftR, rightR = postprocess.rectifyImg(processedImg, imgArea)

    # write the bounding box coordinate
    os.remove(os.path.join(tmpPath, 'matresult', imgNameNoPath+'_blob_0.mat'))
    os.remove(os.path.join(tmpPath, 'ppmImg', imgNameNoPath+'.ppm'))
    os.remove(os.path.join(tmpPath, 'densecrf', imgNameNoPath+'.bin'))
    os.remove(os.path.join(tmpPath, 'resizedImg', imgName))
    result=postprocess.boundingRect(processedImg)
    print(result)
    return result

#imgpath='D:\\ContainerIMG\\2xianghao\\0006\\126_0006_170504_100814_Rear_02.jpg'
# result=detectContainer(imgpath)
# img=cv2.imread(imgpath)
# cv2.rectangle(img, (result[2],result[0]), (result[3],result[1]), (0,255,0), 2)
# cv2.imshow('123',img)
# cv2.waitKey(0)
from socket import *
import json
#import ctime
def deeplab_server():

    #global text_detector
    HOST=''
    PORT=6004
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
            print("error")
            tcpClientSock.close()
            break
        if not data:
            print('not data')
            err_response={"error_code":1}
            response_str=json.dumps(err_response)+'\0'
            tcpClientSock.send(response_str.encode('utf8'))
            break
        imgpath=data.decode('utf8')
        #img=cv2.imread(imgpath,1)
        result=detectContainer(imgpath)
        response={}
        response['y0']=int(result[0])
        response['y1']=int(result[1])
        response['x0']=int(result[2])
        response['x1']=int(result[3])
        #for i in range(text_lines.shape[0]):
        #    response[i]=text_lines[i].tolist()
        response["error_code"]=0
        #response["boxnum"]=text_lines.shape[0]
        response_str=json.dumps(response)+'\0'
        tcpClientSock.send(response_str.encode('utf8'))
        #print([ctime()], ':', imgpath)
        tcpClientSock.close()
    sock.close()
if __name__=='__main__':
    deeplab_server();