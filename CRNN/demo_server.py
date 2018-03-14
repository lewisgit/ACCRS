import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
import models.crnn as crnn
import json

model_path = './models/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

def recog(imgpath):
	global model
	global transformer
	image = Image.open(imgpath).convert('L')
	image = transformer(image)
	if torch.cuda.is_available():
	    image = image.cuda()
	image = image.view(1, *image.size())
	image = Variable(image)

	model.eval()
	preds = model(image)

	_, preds = preds.max(2)
	preds = preds.transpose(1, 0).contiguous().view(-1)

	preds_size = Variable(torch.IntTensor([preds.size(0)]))
	raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
	sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
	print('%-20s => %-20s' % (raw_pred, sim_pred))
	return sim_pred


from socket import *
#import ctime
def crnn_server():
    global predictor
    HOST=''
    PORT=6003
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
        num=0;
        imgfile=str(num)+'.jpg'

        imgpath=imgsavepath+imgfile
        print(imgpath)
        while os.path.exists(imgpath):
        	response_json[num]=recog(imgpath)
        	num+=1
        	imgfile=str(num)+'.jpg'
        	#imgpath=os.path.join(imgsavepath,imgfile)
        	imgpath=imgsavepath+imgfile
        response_json['strnum']=num
        response_json["error_code"]=0;
        response_str=json.dumps(response_json)+'\0'
        tcpClientSock.send(response_str.encode('utf8'))
        #print([ctime()], ':', imgpath)
        tcpClientSock.close()
    sock.close()

if __name__ == '__main__':
    print('server started!')
    #init()
    crnn_server()