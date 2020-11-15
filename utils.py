from mean_average_precision import MeanAveragePrecision
import os
import glob
import PIL
import numpy as np
import pandas as
import math
import cv2
from xml.etree import ElementTree
from contextlib import contextmanager

############ Convenience Functions ########################

#convenience function to build portable paths
join_path = lambda *l: os.sep.join(l)

#convenience function
#used in combination with _with_, sets path as cwd inside with block, and restores previous working dir upon exit
@contextmanager
def cwd(*l):
    path = os.sep.join(l)
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

############ YOLO ########################

def convert_bbox_to_yolo(size, box):      
    '''
    Convert xml BBox to YOLO format

    INPUT
        size: image size (width, height)
        box: box coordinates (xmin, xmax, ymin, ymax)
    OUTPUT
        BBox information encoded for YOLO, (x,y,w,h)
        (x,y): center of the box, rescaled to be within 0 and 1
        (w, h): width and height of BBox, rescaled
    '''                                                                                               
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

    

def convert_annot_yolo(ann_path, detection_classes, outdir=''):
    '''
    Converts annotation file at ann_path into YOLO formt, storing it in outdir
    '''
    img_name,_ = os.path.splitext(os.path.basename(ann_path))
    tree = ElementTree.parse(ann_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in detection_classes:
            print('WARNING: skipped BBox of image %s with undefined class'%(img_name) , cls)
            continue
        cls_id = detection_classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_bbox_to_yolo((w,h), b)
        with open(join_path(outdir, img_name + ".txt"), 'w') as writer:
            writer.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

            
def predict_yolo(net, img_path, net_input_w, net_input_h, **kwargs):
    '''
    INPUT
        net: trained model loaded using opencv
        img_path: path to the image
        net_input_w: network input width (for input layer)
        net_input_h: network input height (for input layer)
        
    OUTPUT
        Returns the bounding boxes as a np.array. Each row is a bounding box, each column is
        (x, y, w/2, h/2, class_id, confidence)
        (x,y): center of the bounding box
        (w,h): width and height of the bounding box
        class_id: numerical id of the class
    '''
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (net_input_w, net_input_h), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    layers_output = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in layers_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width /2)
                h = int(detection[3] * height /2)

                boxes.append([center_x, center_y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    #no prediction
    if len(boxes) == 0:
        return np.zeros((0,6))
    result = np.hstack([np.array(boxes), np.array(class_ids)[:, np.newaxis], np.array(confidences)[:, np.newaxis]])
    ########## TO DELETE! #############
    if result.shape[0] > 4:
        print('POTENTIAL ERROR: n. predicted BBox %d image %s' % (result.shape[0], img_path))
    return result


############ MODEL EVALUATION ########################


#give path if folder structure contains Images and Annotations, else can give img_path and ann_path
def evaluate_model(model, predict_fn, classes, mdl_type='detection', **kwargs):
    '''
    Function used to evaluate model, possibly on test set. Can accept a generic model, coupled with its predict function

    INPUT
        model: a model trained
        predict_fn: custom predict functions for the model with signature _(model, img_path, **kwargs). 
                    Its output can vary based on mdl_type:
                    Detection: Should output a numpy array
                        containing BBoxes on each row, as (x, y, w/2, h/2, class_id, confidence)
                    Classification: NOT IMPLEMENTED
        classes: list of class labels
        mdl_type: kind of problem type: detection or classification 
        **kwargs: 
            path: path to folder containing Images and Annotations folder
            img_path and ann_path: separate paths for the two folders.
            mAP_type: type of mAP metric to use, pascal_voc or coco. Default: pascal_voc
            Additional parameters for predict_fn.
    OUTPUT
        Outputs evaluation metric for the model. Depends on mdl_type:
        Detection: mean average precision (mAP), based on mAP_type.
        Classification: NOT IMPLEMENTED.
    '''
    n_classes = len(classes)
    classes_map = {classes[i].lower() : i for i in range(n_classes)}
    
    
    #input check
    if 'path' in kwargs:
        path = kwargs['path']
        if not (os.path.isdir(join_path(path,'Images')) and os.path.isdir(join_path(path,'Annotations'))):
            raise Exception('Could not find directories Images and Annotations within given path')
        f_img_path = join_path(path,'Images')
        f_ann_path = join_path(path,'Annotations')
    elif ('img_path' in kwargs and 'ann_path' in kwargs):
        f_img_path = kwargs.get('f_img_path')
        f_ann_path = kwargs.get('f_ann_path')
    else:
        raise Exception('You need to supply a path to images and annotations')
        
    metric_fn = MeanAveragePrecision(num_classes=n_classes)
    if mdl_type not in ['detection', 'classification']:
        raise Exception('Unknown model type, must be either detection or classification.')
    
    #in case img_path == ann_path
    for img_path in glob.glob(join_path(f_img_path, '*[!.xml]')):
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        try:
            gt_bboxes = read_xml_bb(join_path(f_ann_path, img_name + '.xml'), classes_map)
            preds = predict_fn(model, img_path, **kwargs)
            if mdl_type == 'detection':
                preds = convert_c_bbox_to_corners(preds)
                gt = np.zeros((gt_bboxes.shape[0], gt_bboxes.shape[1]+1))
                gt[:,:-1] = gt_bboxes
                metric_fn.add(preds, gt)
            elif mdl_type == 'classification':
                raise Exception('not implemented')
                
        except Exception as e:
            print('Found exception processing image %s' % (img_path))
            raise e from None
    
    if mdl_type == 'detection':
        mAP_type = kwargs.get('mAP_type', 'pascal_voc')
        if mAP_type == 'pascal_voc':
            mAP = metric_fn.value(iou_thresholds=0.5)['mAP']
        elif mAP_type == 'coco':
            mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        elif mAP_type == 'both':
            mAP = dict()
            mAP['pascal_voc'] = metric_fn.value(iou_thresholds=0.5)['mAP']
            mAP['coco'] = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        else:
            raise Exception('mean average precision type unknown %s' % mAP_type)
    return mAP
            


############ MISC ########################


def read_xml_bb(ann_path, classes_map):
    '''
    INPUT
        ann_path: path to annotation file (xml)
        classes_map: dictionary containing key=class_label value=number
    OUTPUT
        numpy array containing a BBox for each row, as (xmin, xmax, ymin, ymax, class_id, difficulty)
    '''
    bboxes = []
    tree = ElementTree.parse(ann_path)
    root = tree.getroot()
    for member in root.findall('object'):
        xmlbox = member.find('bndbox')
        value = [
                int(xmlbox.find('xmin').text),
                int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text),
                int(xmlbox.find('ymax').text),
                classes_map[member.find('name').text.lower()],
                int(member.find('difficult').text)]
        bboxes.append(value)
    return np.array(bboxes)

    
def convert_c_bbox_to_corners(boxes):
    '''
    INPUT
        numpy array of bounding boxes, as (x, y, w/2, h/2, ...)
    OUPUT
        numpy array of bounding boxes, as (xmin, xmax, ymin, ymax, ...)
    '''
    xmin = boxes[:,0] - boxes[:,2]
    xmax = boxes[:,0] + boxes[:,2]
    ymin = boxes[:,1] - boxes[:,3]
    ymax = boxes[:,1] + boxes[:,3]
    return np.hstack([xmin[:,np.newaxis], ymin[:,np.newaxis], xmax[:,np.newaxis], ymax[:,np.newaxis], boxes[:,4:]])
    
    
def _convert_img_to_jpg(path):
    '''
    Converts image at path to jpg
    '''
    dir, file = os.path.split(path) 
    img_name,_ = os.path.splitext(file)
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    img.save(join_path(dir, img_name + '.jpg'))
    img.close()


def convert_to_jpg(path):
    '''
    Converts PNG and jpeg images at path to jpg
    '''
    
    #cast png to jpg
    pngs = glob.glob(join_path(path, '*.png'))
    pngs.extend(glob.glob(join_path(path, '*.PNG')))
    for png in pngs:
        _convert_img_to_jpg(png)
        os.remove(png)

    #cast jpeg to jpg
    pngs = glob.glob(join_path(path, '*.jpeg'))
    for png in pngs:
        _convert_img_to_jpg(png)
        os.remove(png)

# converting xmls to df
def xml_to_df(path):
    '''
    Collects all xmls at path and creates a pandas dataframe
    '''
    xml_list = []
    for xml_file in glob.glob(path +'/*.xml'):
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            xmlbox = member.find('bndbox')
            value = (''.join(xml_file.split(os.sep)[-1].split('.')[:-1])+'.jpg',
                     int(xmlbox.find('xmin').text),
                     int(xmlbox.find('ymin').text),
                     int(xmlbox.find('xmax').text),
                     int(xmlbox.find('ymax').text),
                     member.find('name').text.lower())
            xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

            