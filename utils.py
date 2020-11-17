from mean_average_precision import MeanAveragePrecision
import os
import glob
import PIL
import numpy as np
import pandas as pd
import math
import cv2
from xml.etree import ElementTree
from contextlib import contextmanager
import re
import pathlib
import sys
import tensorflow as tf




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

############ SSD #####################################

def xml_to_csv(path):
    
    '''
    This function convert all the .xml file into a single .csv file
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

def label_map(objname, repo):    
    
    '''
    This function crate the label map
    INPUT
        objname: string containing the name of the class
        repo: name of the repositort where we are working
    '''    
    with open(os.path.join(os.getcwd(), repo , 'OD_SSD/label_map.pbtxt'), 'a') as the_file:
        the_file.write('item\n')
        the_file.write('{\n')
        the_file.write('id :{}'.format(int(1)))
        the_file.write('\n')
        the_file.write("name :'{0}'".format(str(objname)))
        the_file.write('\n')
        the_file.write('}\n')

def configuring_pipeline(pipeline_fname,fine_tune_checkpoint, train_record_fname, test_record_fname, label_map_pbtxt_fname, batch_size, num_steps):
    
    '''
    This function modify the config file according to our parameters
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
          '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

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


def predict_fn_ssd(graph, image_path, **kwargs):
    '''
    INPUT
        graph: path to the trained model
        img_path: path to the image
        
    OUTPUT
        Returns the bounding boxes as a np.array. Each row is a bounding box, each column is
        (x, y, w/2, h/2, class_id, confidence)
        (x,y): center of the bounding box
        (w,h): width and height of the bounding box
        class_id: numerical id of the class
    '''

    from object_detection.utils import ops as utils_ops
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
    
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)
    
    image = PIL.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                      'num_detections', 'detection_boxes', 'detection_scores',
                      'detection_classes', 'detection_masks'
                  ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                            real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                            real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                      detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                      tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                      detection_masks_reframed, 0)
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                                  output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                                  'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
                             

            #custom part
            im_height, im_width, _ = image.shape
            d_boxes = output_dict['detection_boxes']
            y = (d_boxes[:,0] + d_boxes[:,2])/2*im_height
            x = (d_boxes[:,1] + d_boxes[:,3])/2*im_width
            half_w = (d_boxes[:,3] - d_boxes[:,1])/2*im_width
            half_h = (d_boxes[:,2] - d_boxes[:,0])/2*im_height
            det_scores = output_dict['detection_scores']
            class_id = output_dict['detection_classes']

            output_matrix = np.dstack((x,y,half_w,half_h, class_id,det_scores))

  
    return np.squeeze(output_matrix)



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

            
