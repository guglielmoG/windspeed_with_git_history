import cv2
import glob
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import PIL
import random
import re
import seaborn as sns
import shutil
import sys
import tarfile
import tensorflow as tf 
import urllib.request

from collections import namedtuple, OrderedDict
from contextlib import contextmanager
from mean_average_precision import MeanAveragePrecision
from xml.etree import ElementTree

from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix

from windspeed.retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


# ########### Convenience Functions ########################

#convenience function to build portable paths
join_path = lambda *l: os.sep.join(l)


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


# ########### YOLO ########################

def convert_bbox_to_yolo(size, box):
    '''
    Converts xml BBox to YOLO format

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
    Converts annotation file at ann_path into YOLO format, storing it in outdir
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
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()
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


############ RETINANET #####################################
def predict_retinanet(net,image):
    '''
    INPUT
        net: trained Retinanet model 
        image: image in BGR format

    OUTPUT
        Returns:
        - the bounding boxes as a np.array. Each row is a bounding box, 
        each column is (x_min, y_min, x_max, y_max)
        scores: confidence of each box
        labels: labels associated to each box
    '''
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = net.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    return boxes, scores, labels

def predict_retinanet_yolo_format(net,img_path,**kwargs):
    '''
    INPUT
        net: trained Retinanet model 
        img_path: path to the image

    OUTPUT
        Returns the bounding boxes as a np.array. Each row is a bounding box, each column is
        (x, y, w/2, h/2, label, score)
        (x,y): center of the bounding box
        (w,h): width and height of the bounding box
        label: numerical id of the class
        score: confidence of the prediction
    '''
    image = read_image_bgr(img_path)
    boxes, scores, labels = predict_retinanet(net,image)
    bboxes=[]
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        x = int((box[0] + box[2])/2.0)
        y = int((box[1] + box[3])/2.0)
        w = int((box[2] - box[0])/2.0)
        h = int((box[3] - box[1])/2.0)

        if score < 0.5:
            break
        bboxes.append([x,y,w,h,label,score])
    if len(bboxes) == 0:
        return np.zeros((0,6))

    return np.array(bboxes)

def predict_image_retinanet(net,label_map,image):
    '''
    INPUT
        net: trained Retinanet model
        label_map: dictionary with possible labels as values and integers as keys
        image: image to process in BGR format

    OUTPUT
        Returns the image with the bounding box
    '''
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    boxes, scores, labels = predict_retinanet(net,image)
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            break   
        
        b = box.astype(int)
        caption = "{} {:.3f}".format(label_map[label], score)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        
    return draw

def predict_video_retinanet(video_path,output_name,model,label_map):
    '''
    INPUT
        video_path: path of the video to process
        output_name: path of the output where to find the processed video 
        model: retinanet model to use
        label_map: dictionary with possible labels as values and integers as keys
    OUTPUT
        Returns the video in the desidered folder
    '''
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res=(int(width), int(height))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(output_name, fourcc, 30.0, res)
        while True:
            try:
                is_success, frame = cap.read()
            except cv2.error:
                continue
            if not is_success:
                break
            draw=predict_image_retinanet(model,label_map,frame)
            image = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            out.write(image)
        out.release() 
    cap.release()
############ SSD #####################################

def xml_to_csv(path):

    '''
    This function converts all the .xml file into a single .csv file
    INPUT
        path: path to the directory where the .xml files are
    '''
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        base = os.path.basename(xml_file)
        file_name, _ = os.path.splitext(base)
        file_name = file_name + '.jpg'
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (#root.find('filename').text,
                     file_name,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def label_map(objname, path_to_dir):
    '''
    This function creates the label map
    INPUT
        objname: string containing the name of the class
        repo: name of the repositort where we are working
    '''
    path_to_pbtxt = os.path.join(path_to_dir, 'label_map.pbtxt')
    with open(path_to_pbtxt, 'a') as the_file:
        the_file.write('item\n')
        the_file.write('{\n')
        the_file.write('id :{}'.format(int(1)))
        the_file.write('\n')
        the_file.write("name :'{0}'".format(str(objname)))
        the_file.write('\n')
        the_file.write('}\n')
    return path_to_pbtxt


def configuring_pipeline(pipeline_fname,fine_tune_checkpoint, train_record_fname, test_record_fname, label_map_pbtxt_fname, batch_size, num_steps):
    '''
    This function modifies the config file according to our parameters
    INPUT
        pipeline_fname: path to the .config file
        fine_tune_checkpoint: path of the pretrained model
        train_record_fname: path to the tran .tfrecord
        test_record_fname: path to the test .tfrecord
        label_map_pbtxt_fname: path to the label map file
        batch_size: this is the batch size with which we want to train our model
        num_sted: number of step with which we want to train our model
    '''
    with open(pipeline_fname) as f:
        s = f.read()
    with open(pipeline_fname, 'w') as f:
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
        # tfrecord files train and test.
        s = re.sub(
          '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
          '(input_path: ".*?)(valid.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)
        # label_map_path
        s = re.sub(
          'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)
        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)
        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)
        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(1), s)
        f.write(s)


def predict_ssd(detect_fn, img_path, **kwargs):

    image_np = np.array(PIL.Image.open(img_path))
    height, width, channels = image_np.shape
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0]
    scores = detections['detection_scores'].numpy()[0]

    y = ((boxes[:,0] + boxes[:,2])/2*height).astype(np.int)
    x = ((boxes[:,1] + boxes[:,3])/2*width).astype(np.int)
    half_w = ((boxes[:,3] - boxes[:,1])/2*width).astype(np.int)
    half_h = ((boxes[:,3] - boxes[:,1])/2*height).astype(np.int)
    classes = classes - 1

    output_matrix = np.dstack((x,y,half_w,half_h, classes,scores))
    output_matrix = np.squeeze(output_matrix)
    output_matrix = output_matrix[output_matrix[:,5]>=0.5,:]

    if len(output_matrix) == 0:
        return np.zeros((0,6)).astype(np.int8)

    return output_matrix


def generate_tfrecord(csv_input, output_path, image_dir):

    from object_detection.utils import dataset_util

    def class_text_to_int(row_label):
        if row_label == 'flag':
        # if row_label == 'tommad':
            return 1
        else:
            return 0

    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


# ########### MODEL EVALUATION ########################

#give path if folder structure contains Images and Annotations, else can give img_path and ann_path
def evaluate_model(model, predict_fn, classes, mdl_type='detection', **kwargs):
    '''
    Function used to evaluate model, possibly on test set. It can accept a generic model, coupled with its predict function

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


# ########### MISC ########################

def read_xml_bb(ann_path, classes_map):
    '''
    INPUT
        ann_path: path to annotation file (xml)
        classes_map: dictionary containing key=class_label value=number
    OUTPUT
        numpy array containing a BBox for each row, as (xmin, ymin, xmax, ymax, class_id, difficulty)
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
        numpy array of bounding boxes, as (xmin, ymin, xmax, ymax, ...)
    '''
    xmin = boxes[:,0] - boxes[:,2]
    xmax = boxes[:,0] + boxes[:,2]
    ymin = boxes[:,1] - boxes[:,3]
    ymax = boxes[:,1] + boxes[:,3]
    return np.hstack([xmin[:,np.newaxis], ymin[:,np.newaxis], xmax[:,np.newaxis], ymax[:,np.newaxis], boxes[:,4:]])


def convert_corners_to_c_bbox(boxes):
    '''
    INPUT
        numpy array of bounding boxes, as (xmin, ymin, xmax, ymax, ...)
    OUPUT
        numpy array of bounding boxes, as (x, y, w/2, h/2, ...)
    '''
    x = (boxes[:,0] + boxes[:,2]) / 2
    y = (boxes[:,1] + boxes[:,3]) / 2
    w_2 = (boxes[:,2] - boxes[:,0]) / 2
    h_2 = (boxes[:,3] - boxes[:,1]) / 2
    return np.hstack([x[:,np.newaxis], y[:,np.newaxis], w_2[:,np.newaxis], h_2[:,np.newaxis], boxes[:,4:]])


def enlarge_boxes(boxes,ratio=1.1,xml=True):
    '''
    enlarges bounding boxes by specified ratio.
    INPUT
        boxes = numpy array of bounding boxes, as
            if xml=True: (xmin, ymin, xmax, ymax, ...)
            if xml=False: (x, y, w/2, h/2, ...)
        ratio = ratio by which the boxes are enlarged
        xml = True if boxes from the annotated xml files, False if boxes from object detection
    OUTPUT
        numpy array of enlarged bounding boxes, as (x, y, w'/2, h'/2, ...)
    '''
    if xml:
        boxes = convert_corners_to_c_bbox(boxes)

    boxes[:,2] *= ratio
    boxes[:,3] *= ratio

    return boxes


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


def annotations_to_df(path,classes_map):
    '''
    Collects all xmls at path and creates a pandas dataframe
    INPUT:
        path: folder path where the annotations are stored
        classes map: dictionary containing key=class_label value=number
    OUTPUT:
        Dataframe that summarize the main feature of every annotation
    '''
    xml_list = []
    for xml_file in glob.glob(path +os.sep+'*.xml'):
        value=read_xml_bb(xml_file,classes_map)[:,:-1]
        names=np.array([[''.join(xml_file.split(os.sep)[-1].split('.')[:-1])+'.jpg']]*value.shape[0])
        annots=np.hstack((names, value))
        for ann in annots:
            xml_list.append(ann)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax','class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df['class']=xml_df['class'].apply(lambda x:list(classes_map.keys())[list(classes_map.values()).index(int(x))])
    return xml_df


# ########### IMAGE CLASSIFICATION #####################################

def get_flags(img_path,boxes,ratio=1.1,xml=True,new_dim=None):
    '''
    extracts only the detected flags from an image, after enlarging the bounding boxes.
    INPUT
        img_path = path to specific image
        boxes = numpy array of bounding boxes, as
            if xml=True: (xmin, ymin, xmax, ymax, ...)
            if xml=False: (x, y, w/2, h/2, ...)
        ratio = ratio by which the boxes are enlarged
        xml = True if boxes from the annotated xml files, False if boxes from object detection (yolo format)
        new_dim = None or target_size
    OUTPUT
        flags = list of cropped images, as numpy arrays
        labels = if xml = True, list flag labels from manual annotation; if xml = False, an empty list.
    '''
    img = load_img(img_path)
    boxes = enlarge_boxes(boxes,ratio,xml)
    boxes = convert_c_bbox_to_corners(boxes)
  
    flags = []
    labels = []

    for box in boxes:
        im = img.crop(box[:4])
        if new_dim is not None:
            im = im.resize(new_dim)
        flags.append(img_to_array(im))
        if xml:
            labels.append(int(box[4]))

    return flags,labels

pattern = '[a-z][a-z][a-z]+'

def get_location_names(annot_path):
    '''
    extracts the names of all locations for the cams annotations
    INPUT
        path to the folder containing the annotatated files
    OUTPUT
        a list of unique locations, sorted alphabetically to ensure reproducibility. 
    '''
    locs = []
    for annot in os.listdir(annot_path):
        loc = re.findall(pattern,annot)[0]
        locs.append(loc)

    return sorted(list(dict.fromkeys(locs)))

def split_train_test_locations(locations,val_split,test_split,seed):
    '''
    performs a random split of train, validation and test set, for the different cams locations
    INPUT
        locations = list of camera locations
        val_split = % split of training and validation
        test_split = % split of training+validation and test
        seed = random seed for reproducibility
    OUTPUT
        sets containing locations allocated to training, validation and test data respectively
    '''
    n_cams = len(locations)
    random.seed(seed)

    train = random.sample(locations,round(n_cams*(1-test_split)))
    validation = set(random.sample(train,round(len(train)*val_split)))
    test = {i for i in locations if i not in train}
    train = {i for i in train if i not in validation}

    return train,validation,test

allowed_extensions = ('.png','.PNG','.jpg','.jpeg')

def create_classification_directory(cams_dir,annot_map,info=True,val_split=0.2,test_split=0.2,seed=3456):
    '''
    creates the correct folder structure to be fed into keras' ImageDataGenerator via flow_from_directory()
    INPUT
        cams_dir = directory to the cams folder, containing Images and Annotations
        annot_map = dictionary mapping string labels to numerical values
        info = specifies whether we want to get printed information on the size and composition of the splits
        val_split = % split of training and validation
        test_split = % split of training+validation and test
        seed = random seed for reproducibility
    OUTPUT
        this function performs these three operations:
        1. creates the subdirectories train, validation and test set inside of cams_dir
        2. crops out the flags from each of the Images using the bounding boxes in Annotations
        3. places the flags into further subdirectories, for each of the three datasets, based on their label
    '''
    if not (os.path.isdir(join_path(cams_dir,'Images')) and os.path.isdir(join_path(cams_dir,'Annotations'))):
        raise Exception('Could not find directories /Images and /Annotations within given path')
    annot_path = join_path(cams_dir,'Annotations')
    img_path = join_path(cams_dir,'Images')

    locations = get_location_names(annot_path)
    tr,val,te = split_train_test_locations(locations,val_split,test_split,seed)

    for i in ['train','validation','test']:
        os.mkdir(join_path(cams_dir,i))
        for v in annot_map.values():
            os.mkdir(join_path(cams_dir,i,str(v)))

    for img in os.listdir(img_path):
        if not img.endswith(allowed_extensions): 
            print(f'WARNING: {img} in /Images was not recognized as an image, and thus ignored')
            print('')
            continue

        loc = re.findall(pattern,img)[0]
        dest = 'test'
        if loc in tr:
            dest = 'train'
        elif loc in val:
            dest = 'validation'

        annot = join_path(annot_path,img[:-4]+'.xml')
        try:
            boxes = read_xml_bb(annot,annot_map)
        except FileNotFoundError:
            print(f'WARNING: xml file for {img} was not found in /Annotations, and thus ignored')
            print('')
            continue
        flags,labels = get_flags(join_path(img_path,img),boxes)

        j = 1 #keeps count of flags in given img
        for f in range(len(flags)):
            PIL.Image.fromarray(flags[f].astype(np.uint8)).save(
                join_path(cams_dir,dest,str(int(labels[f])),img[:-4]+'_'+str(j)+'.png'),format='PNG')
            j+=1

    if info:
        train_dirs = glob.glob(join_path(cams_dir,'train/*'))
        val_dirs = glob.glob(join_path(cams_dir,'validation/*'))
        test_dirs = glob.glob(join_path(cams_dir,'test/*'))
        
        print(f'label map: {annot_map}')
        print(f'total training images by label: {[len(os.listdir(k)) for k in train_dirs]}')
        print(f'total validation images by label: {[len(os.listdir(k)) for k in val_dirs]}')
        print(f'total test images by label: {[len(os.listdir(k)) for k in test_dirs]}')


def plot_conf_mat(y_true,y_pred,labels,normalize=False,cmap=sns.cm.rocket_r,figsize=(10,7)):
    '''
    plots confusion matrix, relying on sklearn and seaborn 
    '''
    cm = confusion_matrix(y_true,y_pred)
    fmt = ".0f"
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"

    plt.figure(figsize=figsize)
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=labels, yticklabels=labels, annot_kws={"size": 16},cmap=cmap)
    plt.show()


# ########### FINAL PREDICTION #####################################

def split_train_test_locations_df(cams_dir,df,val_split=0.3,test_split=0.2,seed=3456):
    '''
    Performs allocation of each row between train, validation and test set, randomly by location.
    INPUT
        cams_dir: directory containing the df and subfolders /Images and /Annotations
        df = DataFrame with columns 'image_id' and 'true_label'
        val_split = % split between train and validation
        test_split = % split between train+validation and test
        seed = for reproducibility
    OUTPUT
        adds two new columns to df:
        df['split'] = whether that image belongs to train, test or validation
        df['location'] = location of the image, useful for diagnostics
    '''
    annot_path = join_path(cams_dir,'Annotations')
    locations = get_location_names(annot_path)
    tr,val,te = split_train_test_locations(locations,val_split,test_split,seed)
    ids = df.image_id.tolist()
    split = []
    locs = []
    for i in range(len(ids)):
        loc = re.findall(pattern,ids[i])[0]
        locs.append(loc)
        dest = 'test'
        if loc in tr:
            dest = 'train'
        elif loc in val:
            dest = 'validation'
        split.append(dest)
    df['split'] = split
    df['location'] = locs
    
    return df


def img_pred_2step(cams_dir,img, retina_model, effnet_model,th=0.5):
    '''
    Predicts wind intensity for the image overall
    Seeks flags, and based on the flags predicts the intensity. Returns -1 if no flags are found.
    INPUT:
        cams_dir: directory containing the df and subfolders /Images and /Annotations
        img: filename of image
        retina_model: LOADED retinanet model from .h5 file
        effnet_model: LOADED efficientnet model from .h5 file
        th: acceptance threshold for flag detection algorithm
        label_map: label mapping for classification
    OUTPUT:
        final prediction for the image (as -1, 0, 1, 2)
    '''
    try:
      img_path = join_path(cams_dir,'Images',img)
    except:
        print('No image with this filename was found in \Images')
        return
  
    image = read_image_bgr(img_path)
    try:
      boxes,scores,_ = predict_retinanet(retina_model, image)
    except AttributeError:
      print('WARNING: please load your trained detection model first, using models.load_model()')
      return 
    
    #get only boxes containing flags with confidence >= th 
    mask = scores[0] > th
    boxes = boxes[0][mask]
    
    #if no flags found
    if boxes.shape[0] == 0:
        return -1
    
    #grab flags
    flags, _ = get_flags(img_path, boxes,xml=False,new_dim=(240,240))

    #run classifier 
    out_class = np.zeros((len(flags), 3))
    for i, flag in enumerate(flags):
        try:
          preds = effnet_model.predict(np.expand_dims(flag, axis=0))
        except AttributeError:
          print('WARNING: please load your trained classification model first, using load_model()')
          return 
        out_class[i,:] = preds
    
    # combine output
    avg = np.mean(out_class, axis=0)
    final_pred = np.argmax(avg)
            
    return final_pred



