<h1>DISCLAIMER</h1>

<h2>This model is not approved by any institution or organization, it's only to test how deep learning works detecting the subfamily Orthocoronavirinae(Coronavirus), that belongs to the family Coronaviridae.</h2>

<h2>This model does not detect COVID-19, it detects the subfamily Orthocoronavirinae(Coronavirus virion).</h2>

<hr>


<img src="https://github.com/Ariel0123/Orthocoronavirinae-Coronavirus/blob/master/output_test/images_test/image10.jpg" />

<h4>Installation</h4>
<strong>Create environment -Optional-</strong><br>

```python
virtualenv NAME-ENVIROMENT
```

<strong>Activate enviroment</strong>

```python
source NAME-ENVIROMENT/bin/activate
```

<strong>Install requirements</strong>

```python
pip install -r requirements.txt
```

<h4>Usage</h4>
<strong>Run the program</strong><br>

```python
python detection.py
```
<strong>The "images_test" folder is where the images to detect are located.</strong><br>
<strong>In the folder "output_test" is where the results of the images are.</strong><br>

<strong>If the images are in png or jpeg format, they will be automatically converted to jpg to avoid errors.</strong>

<strong>The model was trained with TensorFlow Object Detection API, the model is the "faster_rcnn_inception_v2_coco".</strong>


