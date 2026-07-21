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
txt = f.read_text()
txt = txt.replace('    if timeseries:\n',
                  '    if timeseries is not None:\n')
# Fix 4: gwpy 4.x's Series.crop() dropped the (long-unused) verbose kwarg
# entirely — passing it at all raises TypeError, even verbose=False.
txt = txt.replace('data = timeseries.crop(start_time, stop_time, verbose=verbose)',
                  'data = timeseries.crop(start_time, stop_time)')
f.write_text(txt)

# Fix 5: keras.utils.np_utils removed (Keras 1/2-only name). train_classifier.py
# is imported transitively via gravityspy.table.Events even when only classify()
# is used, so this import runs unconditionally regardless of whether make_model()
# is ever called. to_categorical's behavior is unchanged across Keras versions,
# so plain tensorflow.keras (not tf_keras) is fine here — unlike Fix 2, there's no
# legacy .h5-loading concern.
f = sp / 'ml/train_classifier.py'
f.write_text(f.read_text().replace(
    'from keras.utils import np_utils',
    'from tensorflow.keras.utils import to_categorical as _to_categorical\n'
    'class np_utils:\n'
    '    to_categorical = staticmethod(_to_categorical)'))

# Fix 6: matplotlib >=3.3 renamed LogScale's basex/basey kwargs to base.
# gravityspy/plot/plot.py still calls set_yscale('log', basey=2) in two places
# (per-class Q-scan plots + the summary grid), both hit during every classify().
f = sp / 'plot/plot.py'
f.write_text(f.read_text().replace("basey=2", "base=2"))

# Fix 7: gravityspy/ml/read_image.py predates numpy>=1.24 and scikit-image's
# rescale() API change — skimage renamed multichannel=bool to channel_axis
# (None = single-channel, -1 = channel on the last axis), and np.int (a bare
# alias for the builtin int) was removed from numpy entirely. Both functions
# (read_gray, read_rgb) are hit depending on GravitySpy's configured image mode.
f = sp / 'ml/read_image.py'
txt = f.read_text()
txt = txt.replace("preserve_range='True', multichannel=False",
                  "preserve_range=True, channel_axis=None")
txt = txt.replace("preserve_range='True', multichannel=True",
                  "preserve_range=True, channel_axis=-1")
txt = txt.replace("np.int(", "int(")
f.write_text(txt)

print('All patches applied')
