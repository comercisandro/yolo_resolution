import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import struct
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model


import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.patches import Rectangle



import cv2
from PIL import Image


from pathlib import Path

import tensorflow as tf

import gc

print("Numero de GPUs disponibles: ", len(tf.config.list_physical_devices('GPU')))

tf.config.list_physical_devices(device_type='GPU')


# Carga del modelo yolov3 
model = load_model('model/model.h5')

# Tamaño de entrada del modelo
input_w, input_h = 416, 416


#Funcion para obtener las BoundBox
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
 
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
 
        return self.label
 
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
 
        return self.score


#Funcion sigmoide
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


#Decodificacion de la red neuronal
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


#Correccion de las bbox
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
 

#Puntos en bbox
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
        

#Tamaños
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    

#Cargar y preparar la imagen
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img('input/'+filename)
    imagen = plt.imread('input/'+filename)
    
    width, height = image.size
    # load the image with the required size
    image = load_img('input/'+filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    
    
    plt.imsave(procesadas+filename, imagen)
    
    
    
    return image, width, height


#Contar la cantidad de objetos
def count_objects(v_labels):
    labels = {}
    
    for class_name in v_labels:
        
        if class_name in labels:
            
            labels[class_name] = labels[class_name] + 1
            
        
        else:
            labels[class_name] = 1
    
            
            
                   
    return labels


# Obtencion de todos los resultados por arriba de la probabilidad de corte
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# Dibujar los resultados
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread('input/'+filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        
        etiq=count_objects(v_labels)
        
        pyplot.text(x1, y1, label, color='white')
        
    ax.axis('on')
    # save image
    output = plt.gcf()
    output.savefig('procesando.jpg',bbox_inches='tight')
    plt.figure()
    # show the plot
    #pyplot.show()
    
    
    
    
    
def crop_objects(filename, formato, v_boxes, v_labels):
    
    
    
    
    #--------------------------------------------------------------
                      #Hiperparametros
        #--------------------------------------------------------------
        
    movhb=0
    movwb=0
    
    movhi=0
    movwi=0
    
    
    
    
    
    
    #--------------------------------------------------------------
                      #Carga de imagenes y boxes
        #--------------------------------------------------------------
    
    image = pyplot.imread('input/'+filename)
    labels= count_objects(v_labels)
    
    dimension=image.shape
    
    
    
    #number_boxes=1
    
    
    #for i in range(len(v_boxes)):
     #   number_boxes+=1
        
        
        
    #fig,axes = plt.subplots(nrows = number_boxes, ncols = 1, figsize=(10,10))
    
    minhmin=100000
    minwmin=100000
    maxhmax=0
    maxwmax=0
    maxscore=0
    
    direct=filename.split('.')
    
    
    path='output/'
    
    #Path(path+direct[0]+'/'+formato).mkdir(parents=True, exist_ok=True)
    #Path(path+direct[0]).mkdir(parents=True, exist_ok=True)

    
    #for ax in axes.flatten():
     #   ax.axis('off')
    
    
    #--------------------------------------------------------------
                      #Corte de objetos en proporcion y resize
        #--------------------------------------------------------------
    if len(v_boxes)==0 or len(v_labels)==0 or  len(v_scores)==0:
        
        
        hmin=int(dimension[0]/2)-539
        
        hmax=int(dimension[0]/2)+539
        
        wmin=int(dimension[1]/2)-639
        
        wmax=int(dimension[1]/2)+639
        
        w=wmax-wmin

        wmid=wmax-int(w/2)

        h=hmax-hmin

        hmid=int(h/2)

        imgname= 'interseccion.jpg'
        
        #print(hmin,hmax,wmin,wmax)
        
        resize_object(hmin,hmax, hmid, wmid,wmin,wmax,h,w,filename,formato,imgname)

    
    
    else:
            
        for i in range(len(v_boxes)):
            box = v_boxes[i]
            
            label=v_labels[i]
            score=v_scores[i]
            
            

            # get coordinates
            hmin, wmin, hmax, wmax = box.ymin, box.xmin, box.ymax, box.xmax

            if hmin<minhmin:
                minhmin=hmin

            if wmin<minwmin:
                minwmin=wmin

            if hmax>maxhmax:
                maxhmax=hmax

            if wmax>maxwmax:
                maxwmax=wmax
            
            
            
            if label=='person':
                score+=10 
           
            if score>maxscore:
                maxscore=score
                maxscore_position=i
                
        
        
        box_max_prob=v_boxes[maxscore_position]
        label_max_prob=v_labels[maxscore_position]

        
        
        if label_max_prob=='person' and maxwmax-minwmin>1900:
            
            hmin, wmin, hmax, wmax = box_max_prob.ymin, box_max_prob.xmin, box_max_prob.ymax, box_max_prob.xmax

            w=wmax-wmin

            wmid=wmax-int(w/2)

            h=hmax-hmin

            hmid=int(h/2)

            imgname= 'interseccion.jpg'

            
            #resize_object(hmin,hmax, hmid, wmid,wmin,wmax,h,w,filename,formato,imgname)

    #--------------------------------------------------------------
                    #Corte de la interseccion
        #--------------------------------------------------------------

     
        else:
            
            wmin= int(minwmin)-movwi

            wmax= int(maxwmax)-movwi

            hmin= int(minhmin)-movhi

            hmax= int(maxhmax)-movhi


            w=wmax-wmin

            wmid=wmax-int(w/2)

            h=hmax-hmin

            hmid=int(h/2)

            imgname= 'interseccion.jpg'

            #print('wmin',wmin,'wmax',wmax,'wmid',wmid)

        resize_object(hmin,hmax, hmid, wmid,wmin,wmax,h,w,filename,formato,imgname)
        


def resize_object(hmin, hmax, hmid, wmid,wmin, wmax, h, w,filename,formato, imgname):
    
    
    image = pyplot.imread('input/'+filename)
    
    inter= cv2.INTER_AREA
    
    direct=filename.split('.')
    
    path='output/'
    
    #Path(path+direct[0]+'/'+formato).mkdir(parents=True, exist_ok=True)
    
    #Path(path+direct[0]).mkdir(parents=True, exist_ok=True)
    
    dimension=image.shape
    
    
    
    if dimension==(1080, 1920, 3):
    
        if formato=='H4' or formato=='H2':
    
            h=1080
            w=int(h*1.33)

            hmin=0
            hmax=1080
            
            #if wmax-wmin>1300 and detect==True:
             #   wmin=wmin
              #  wmax=int(wmin+w)
            
            #else:
            wmin=(wmid-int(w/2))
            wmax=(wmid+int(w/2))
            
                
            
            #print(wmin,wmax)

            #print('h4 h2',wmin,wmax,hmin,hmax, wmid,w)
            # Si el ancho del objeto estan fuera de la imagen

            if wmin<0:

                wmax=w 
                wmin=0
   
                
            if wmax>1920:
                
                wmin=1920-w
                wmax=1920
                
                
                
            #Si el alto es mas mayor a 1080, el nuevo alto es 1080 
            #y el nuevo ancho es la propocion correcta de 1080 centrado en 1920/2

            if h>1080:
                hmin=0
                hmax=1080

                w=int(1080*1.33)

                wmin=int(1920/2)-int(w/2)
                wmax=int(1920/2)+int(w/2)

            #print('h4 h2',wmin,wmax,hmin,hmax)
            
            cropped_img = image[int(hmin):int(hmax), int(wmin):int(wmax)]



            if formato=='H4':

                dim=(1300,975)

                resized = cv2.resize(cropped_img, dim, interpolation = inter)

                path=('output/H4/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)

            if formato=='H2':

                dim=(800,600)
                resized = cv2.resize(cropped_img, dim, interpolation = inter)
                
                path=('output/H2/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)


            
            


        if formato=='H3' or formato=='H1':

            h=1080
            w=int(h*1.77)

            hmin=0
            hmax=1080
            
            wmin=(wmid-int(w/2))
            wmax=(wmid+int(w/2))
            
            #print(h,w)

            #print('h3 h1',wmin,wmax,hmin,hmax)
            # Si el ancho del objeto estan fuera de la imagen

            if wmin<0:

                wmax=w 
                wmin=0
                
                
                
                
            if wmax>1920:
                
                wmin=1920-w
                wmax=1920

            #Si el alto es mas mayor a 1080, el nuevo alto es 1080 
            #y el nuevo ancho es la propocion correcta de 1080 centrado en 1920/2

            if h>1080:
                hmin=0
                hmax=1080

                w=int(1080*1.77)

                wmin=int(1920/2)-int(w/2)
                wmax=int(1920/2)+int(w/2)


            #print('h3 h1',wmin,wmax,hmin,hmax)
            cropped_img = image[int(hmin):int(hmax), int(wmin):int(wmax)]




            ###print('hmin',hmin,'hmax',hmax,'wmin',wmin,'wmax',wmax)

            if formato=='H3':
                

                dim=(1300,731)
                resized = cv2.resize(cropped_img, dim, interpolation = inter)
                
                path=('output/H3/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)

            if formato=='H1':
                

                dim=(800,450)
                resized = cv2.resize(cropped_img, dim, interpolation = inter)
                
                path=('output/H1/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)

            



        if formato=='H':

            h=1080
            w=int(h*1.75)

            hmin=0
            hmax=1080
            
            wmin=(wmid-int(w/2))
            wmax=(wmid+int(w/2))
            
            #print(h,w)

        
            # Si el ancho del objeto estan fuera de la imagen

            if wmin<0:

                wmax=w 
                wmin=0
                
                
                
                
            if wmax>1920:
                
                wmin=1920-w
                wmax=1920

            #Si el alto es mas mayor a 1080, el nuevo alto es 1080 
            #y el nuevo ancho es la propocion correcta de 1080 centrado en 1920/2

            if h>1080:
                hmin=0
                hmax=1080

                w=int(1080*1.75)

                wmin=int(1920/2)-int(w/2)
                wmax=int(1920/2)+int(w/2)




            cropped_img = image[int(hmin):int(hmax), int(wmin):int(wmax)]

            dim=(900,512)
            resized = cv2.resize(cropped_img, dim, interpolation = inter)

            path=('output/H/')
            Path(path).mkdir(parents=True, exist_ok=True)
            
            plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)



        if formato=='HL':
            

            h=1080
            w=int(h*1.47)

            hmin=0
            hmax=1080
            
            #if wmax-wmin>800 and detect==True:
             #   wmin=wmin
              #  wmax=int(wmin+w)
            
            #else:
            wmin=(wmid-int(w/2))
            wmax=(wmid+int(w/2))
            
            #print(h,w)

        
            # Si el ancho del objeto estan fuera de la imagen

            if wmin<0:

                wmax=w 
                wmin=0
                
                
                
                
            if wmax>1920:
                
                wmin=1920-w
                wmax=1920

            #Si el alto es mas mayor a 1080, el nuevo alto es 1080 
            #y el nuevo ancho es la propocion correcta de 1080 centrado en 1920/2

            if h>1080:
                hmin=0
                hmax=1080

                w=int(1080*1.47)

                wmin=int(1920/2)-int(w/2)
                wmax=int(1920/2)+int(w/2)




            cropped_img = image[int(hmin):int(hmax), int(wmin):int(wmax)]



            dim=(800,542)
            resized = cv2.resize(cropped_img, dim, interpolation = inter)

            path=('output/HL/')
            Path(path).mkdir(parents=True, exist_ok=True)
            
            plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)
        
            
            
        elif formato=='V4' or formato=='V2' or formato=='V3' or formato=='V1' or formato=='V':
            pass

    elif dimension==(1920, 1280, 3):
    
        if formato=='V4' or formato=='V2':
            
            #wmid=int(w/2)
            #w=int(h*0.75)
            #wh4mid=wmid

            #h=1080
            #w=int(h*0.75)
            
            w=1280
            h=int(w*1.33)
            
            wmin=0
            wmax=1280

            hmin=hmid-int(h/2)
            hmax=hmid+int(h/2)
            
            #A la izquierda    
            #wmin=wmin
            #wmax=wmin+w


            # Si el ancho del objeto estan fuera de la imagen

            if hmin<0:

                hmax=h 
                hmin=0




            if hmax>1920:

                hmin=1920-h
                hmax=1920




            cropped_img = image[int(hmin):int(hmax), int(wmin):int(wmax)]



            if formato=='V4':

                dim=(975,1300)

                resized = cv2.resize(cropped_img, dim, interpolation = inter)

                path=('output/V4/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)

            if formato=='V2':

                dim=(542,723)
                resized = cv2.resize(cropped_img, dim, interpolation = inter)

                path=('output/V2/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)







        if formato=='V3' or formato=='V1':
            w=1280
            h=int(w*1.5)
            
            wmin=0
            wmax=1280

            hmin=hmid-int(h/2)
            hmax=hmid+int(h/2)
            
            #A la izquierda    
            #wmin=wmin
            #wmax=wmin+w


            # Si el ancho del objeto estan fuera de la imagen

            if hmin<0:

                hmax=h 
                hmin=0




            if hmax>1920:

                hmin=1920-h
                hmax=1920




            
            cropped_img = image[int(hmin):int(hmax), int(wmin):int(wmax)]


            if formato=='V3':

                dim=(867,1300)

                resized = cv2.resize(cropped_img, dim, interpolation = inter)
                
                path=('output/V3/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)


            if formato=='V1':

                dim=(542,813)
                resized = cv2.resize(cropped_img, dim, interpolation = inter)

                path=('output/V1/')
                Path(path).mkdir(parents=True, exist_ok=True)
                
                plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)





        if formato=='V':

            w=1280
            h=int(w*1.47)
            
            wmin=0
            wmax=1280

            hmin=hmid-int(h/2)
            hmax=hmid+int(h/2)
            
            #A la izquierda    
            #wmin=wmin
            #wmax=wmin+w


            # Si el ancho del objeto estan fuera de la imagen

            if hmin<0:

                hmax=h 
                hmin=0




            if hmax>1920:

                hmin=1920-h
                hmax=1920

            cropped_img = image[int(hmin):int(hmax), int(wmin):int(wmax)]

            dim=(542,800)

            resized = cv2.resize(cropped_img, dim, interpolation = inter)



            path=('output/V/')
            Path(path).mkdir(parents=True, exist_ok=True)
            
            plt.imsave(path+direct[0]+'_'+formato+'_'+imgname, resized)  
            
            
        elif formato=='H4' or formato=='H2' or formato=='H3' or formato=='H1' or formato=='H' or formato=='HL':
            pass  
 
 
 
 
import time
inicio = time.time()

path=('output/')
Path(path).mkdir(parents=True, exist_ok=True)

procesadas=('procesadas/')
Path(procesadas).mkdir(parents=True, exist_ok=True)



# Código a medir
time.sleep(1)
# -------------

formatos=['H4','H3','H2','H1','H','HL','V4','V3','V2','V1','V']


# define our new photo

inputs=os.listdir(path='input')


contador_file=0
contador=0

outputs=os.listdir(path='output/')

cantidad_archivos=0

for i in range(len(inputs)):
    cantidad_archivos+=1

print('Cantidad de imagenes a procesar:', cantidad_archivos)


for i in inputs:
    
    img_name=str(i)
    
    folder=i.split('.')
    
    if folder[0] in outputs:
        
        pass
    
    else:
    
        contador_file+=1

        photo_filename = i

        print('Nombre de la imagen:', i)
        #print(contador_file)

        #photo_filname=path+filename

        # load and prepare image
        image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
        
        # make prediction
        yhat = model.predict(image)
        # summarize the shape of the list of arrays
        #print([a.shape for a in yhat])
        # define the anchors
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        # define the probability threshold for detected objects
        class_threshold = 0.7
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # suppress non-maximal boxes
        do_nms(boxes, 0.5)
        # define the labels
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        # get the details of the detected objects
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
        
        
        if len(v_boxes)==0 or len(v_labels)==0 or  len(v_scores)==0:
            print('No se detectaron objetos en la imagen:', photo_filename)
        
            
            
        # summarize what we found
        #for i in range(len(v_boxes)):
            
            
            #print('Objeto detectado:',v_labels[i],'Probabilidad del objeto', v_scores[i])

        # draw what we found
        draw_boxes(photo_filename, v_boxes, v_labels, v_scores)

        count_objects(v_labels)

        #cortada=crop_objects(photo_filename,'H4', v_boxes, v_labels)

        #fig,axes = plt.subplots(nrows = len(formatos), ncols = 1, figsize=(15,15))




        for i in formatos:

            cortada=crop_objects(photo_filename,i , v_boxes, v_labels)
            

            #axes[contador].imshow(cortada);

            contador+=1
        
        print('Procesada con exito')
        
        cantidad_archivos-=1
        
        print('Imagenes restantes', cantidad_archivos)
        
        os.remove('input/'+img_name)
        
        
        del boxes
        del labels
        del yhat
        del anchors
        
        del photo_filename
        del img_name
        del image
        
        del image_w
        del image_h
        del cortada
        
        del v_boxes
        del v_labels
        del v_scores
        gc.collect()

    


fin = time.time()
print('Tiempo de ejecucion', (fin-inicio)/60 ) 