import os
import sys
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from utils import visualization_utils as vis_util
from utils import label_map_util


MODEL = 'model/frozen_inference_graph.pb'
LABELS = 'config/label_map.pbtxt'
TEST_IMAGES = 'images_test'


MAX_NUMBER_OF_BOXES = 100
MINIMUM_CONFIDENCE = 0.1


label_map = label_map_util.load_labelmap(LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def detect_objects(image_path):
    image = Image.open(image_path)
    image_pro = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_pro, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_pro,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=2)
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(image_pro, aspect = 'auto')
    plt.savefig('output_test/{}'.format(image_path), dpi=72)
    print(boxes)
    plt.close(fig)
    
    
    
images_list = os.listdir(TEST_IMAGES)
for img in images_list:
    if img.lower().endswith(('.png','.jpeg')):
        
        name, ext = img.split('.')
        print(name)
        new_jpg = Image.open(TEST_IMAGES+"/"+img)
        new_jpg = new_jpg.convert('RGB')
        new_jpg.save(TEST_IMAGES+"/"+name+".jpg")
        

IMAGES_READY = glob.glob(os.path.join(TEST_IMAGES, '*.jpg'))
print(IMAGES_READY)

# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for image_path in IMAGES_READY:
            detect_objects(image_path)
