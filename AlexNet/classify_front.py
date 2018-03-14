
import sys
sys.path.append('../../CaffeDeeplab')
import caffe
import os
import scipy.misc
import numpy as np 
#import matplotlib.pyplot as plt

#Global Variables

caffe.set_mode_cpu()
#caffe_root='/home/lewis/caffe-opencv-python'

model_def='deploy.prototxt'
mean_file='mean.npy'
model_weights = 'exVersion_iter_30000.caffemodel'

#save_path='D:/frontChars'

net=caffe.Net(model_def,model_weights,caffe.TEST)
mu=np.load(mean_file)
mu=mu.mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data',mu)
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,0,1))
net.blobs['data'].reshape(1,3,227,227)
def recog(imgsavepath):
	global net
	global transformer
	outstr=''
	#charname=['c1','c2','c3','c4','n1','n2','n3','n4','n5','n6','crc','x1','x2','x3','x4']
	strtable=['0','1','2','3','4','5','6','7','8','9','0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	index=0;
	num=0;
	imgfile=str(num)+'.jpg'
	imgpath=imgsavepath+imgfile
	while os.path.exists(imgpath)
		num+=1
		img_path=imgsavepath+filename+'.jpg'
		image = caffe.io.load_image(img_path)
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image

		#print transformed_image.shape
		#transformed_image[transformed_image>0.5]=255*0.00390625
		#transformed_image[transformed_image<0.5]=0
		#scipy.misc.imsave(save_path+'/out.jpg', transformed_image)
		#plt.imshow(transformed_image)

		output = net.forward()
		output_prob = output['prob'][0]
		outstr=outstr+strtable[output_prob.argmax()]
		#print 'predicted class is:', output_prob.argmax()\
	print outstr

def alexnet_server():
    HOST=''
    PORT=6005
    BUFSIZ=1024
    ADDR=(HOST, PORT)
    sock=socket(AF_INET, SOCK_STREAM)

    sock.bind(ADDR)
    #dicts={'1':2,'2':23,'3':213,'4':500}
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
        imgsavepath=data.decode('utf8')
        #img=cv2.imread(input_path,1)
        #img=cv2.resize(img_org,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        response_json={}
        response_json['result']=recog(imgsavepath)
        #response_json['strnum']=num
        response_json["error_code"]=0;
        response_str=json.dumps(response_json)+'\0'
        tcpClientSock.send(response_str.encode('utf8'))
        #print([ctime()], ':', imgpath)
        tcpClientSock.close()
    sock.close()

if __name__='__main__':
	alexnet_server()