import site, pathlib

sp = pathlib.Path(site.getsitepackages()[0]) / 'gravityspy'

f = sp / 'ml/labelling_test_glitches.py'
txt = f.read_text()
txt = txt.replace('from scipy.misc import imresize',
                  'from skimage.transform import resize as imresize')
txt = txt.replace('from keras.applications.vgg16 import preprocess_input',
                  'from tf_keras.applications.vgg16 import preprocess_input')
txt = txt.replace('from keras.models import load_model',
                  'from tf_keras.models import load_model')
txt = txt.replace('from keras import backend as K',
                  'from tf_keras import backend as K')
f.write_text(txt)

f = sp / 'utils/utils.py'
f.write_text(f.read_text().replace(
    '    if timeseries:\n',
    '    if timeseries is not None:\n'))

print('All patches applied')
