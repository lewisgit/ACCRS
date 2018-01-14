#!/usr/bin/env python3

import os,sys

import time
import datetime
import cv2
import numpy as np
import uuid
import json
import logging
import collections
import tensorflow as tf
from east import model
from east.icdar import restore_rectangle
from east.eval import resize_image, sort_poly, detect
#imp.reload(sys)
#sys.path.append('D:/Lewis/projects/container_ocr/EAST-master/east')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sess, f_score, f_geometry, input_images, global_step = None, None, None, None, None

def get_predictor(checkpoint_path):
    logger.info('loading model')
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)
    return sess, f_score, f_geometry, input_images, global_step

def predictor(img):
    """
    :return: {
        'text_lines': [
            {
                'score': ,
                'x0': ,
                'y0': ,
                'x1': ,
                ...
                'y3': ,
            }
        ],
        'rtparams': {  # runtime parameters
            'image_size': ,
            'working_size': ,
        },
        'timing': {
            'net': ,
            'restore': ,
            'nms': ,
            'cpuinfo': ,
            'meminfo': ,
            'uptime': ,
        }
    }
    """
    start_time = time.time()
    rtparams = collections.OrderedDict()
    rtparams['start_time'] = datetime.datetime.now().isoformat()
    rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
    timer = collections.OrderedDict([
        ('net', 0),
        ('restore', 0),
        ('nms', 0)
    ])

    im_resized, (ratio_h, ratio_w) = resize_image(img)
    rtparams['working_size'] = '{}x{}'.format(
        im_resized.shape[1], im_resized.shape[0])
    start = time.time()
    score, geometry = sess.run(
        [f_score, f_geometry],
        feed_dict={input_images: [im_resized[:,:,::-1]]})
    timer['net'] = time.time() - start

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
    logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
        timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

    if boxes is not None:
        scores = boxes[:,8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    duration = time.time() - start_time
    timer['overall'] = duration
    logger.info('[timing] {}'.format(duration))

    text_lines = []
    if boxes is not None:
        text_lines = []
        for box, score in zip(boxes, scores):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)
    ret = {
        'text_lines': text_lines,
        'rtparams': rtparams,
        'timing': timer,
    }        
    return ret
    


### the webserver
#from flask import Flask, request, render_template
import argparse


class Config:
    SAVE_DIR = 'static/results'


config = Config()


#app = Flask(__name__)

'''
@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')
'''

def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def save_result(img, rst):
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    return rst



#checkpoint_path = '/Users/aub3/Dropbox/DeepVideoAnalytics/shared/east/' if sys.platform == 'darwin' else '/root/model/'

checkpoint_path = 'D:/Lewis/projects/container_ocr/EAST-master/east_icdar2015_resnet_v1_50_rbox'
input_path = 'D:/OneDrive/Lewis/projects/BoxNumberDetect/imgs/126_0062_170214_165717_Rear_03.jpg'

'''
@app.route('/', methods=['POST'])
def index_post():
    global predictor
    global sess
    global f_score
    global f_geometry
    global input_images
    global global_step
    import io
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    if sess is None:
        sess, f_score, f_geometry, input_images, global_step = get_predictor(checkpoint_path)
    rst = predictor(img)
    save_result(img, rst)
    return render_template('index.html', session_id=rst['session_id'])
'''

def demo_main():
    global checkpoint_path
    global input_path
    global predictor
    global sess
    global f_score
    global f_geometry
    global input_images
    global global_step
    import io
    parser = argparse.ArgumentParser()
    #parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--checkpoint-path', default=checkpoint_path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--input_path', default=input_path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    input_path = args.input_path


    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))
    if not os.path.exists(args.input_path):
        raise RuntimeError(
            'input_path `{}` not found'.format(args.input_path))
    
    img_org=cv2.imread(input_path,1)
    #img=cv2.resize(img_org,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    if sess is None:
        sess, f_score, f_geometry, input_images, global_step = get_predictor(checkpoint_path)
    rst = predictor(img)
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
    save_result(img, rst)    
    #app.debug = args.debug
    #app.run('0.0.0.0', args.port)

def init(trained_file):
    global checkpoint_path
    global input_path
    global predictor
    global sess
    global f_score
    global f_geometry
    global input_images
    global global_step
    import io
    if sess is None:
        sess, f_score, f_geometry, input_images, global_step = get_predictor(trained_file)

def detect1(imgpath):
    global checkpoint_path
    global input_path
    global predictor
    global sess
    global f_score
    global f_geometry
    global input_images
    global global_step
    import io
    img=cv2.imread(imgpath,1)
    rst = predictor(img)
    textlines=[]
    for t in rst['text_lines']:
        textlines.append([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'], t['y2'], t['x3'], t['y3']])
    nptext_lines=np.array(textlines)
    return [nptext_lines,]

def init_east():
    global predictor
    global sess
    global f_score
    global f_geometry
    global input_images
    global global_step
    import io
    if sess is None:
        sess, f_score, f_geometry, input_images, global_step = get_predictor(checkpoint_path)

from socket import *
from time import ctime
import json
def east_server():
    global predictor
    HOST=''
    PORT=6001
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
        #img=cv2.resize(img_org,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        rst = predictor(img)
        index=0
        response_json={}
        for t in rst['text_lines']:
            response_json[index]=[t['x0'], t['y0'], t['x1'], t['y1'], t['x2'], t['y2'], t['x3'], t['y3']]
            index+=1
        response_json["error_code"]=0;
        response_json["boxnum"]=index;
        response_str=json.dumps(response_json)+'\0'
        tcpClientSock.send(response_str.encode('utf8'))
        print([ctime()], ':', imgpath)
        tcpClientSock.close()
    sock.close()

if __name__ == '__main__':
    init_east()
    print('server started!')
    east_server()

