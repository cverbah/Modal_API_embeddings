import pandas as pd
import numpy as np
import time
import re
import unidecode
import itertools
from dotenv import load_dotenv
import os
import mysql.connector
import tensorflow as tf
from transformers import TFViTModel, ViTImageProcessor
from tensorflow.python.client import device_lib
from vertexai.preview.language_models import TextEmbeddingModel
from PIL import Image as PIL_Image
from collections import Counter
import cv2
import io
import requests
import warnings
import math
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


# envs
load_dotenv()
MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
MYSQL_HOST = os.environ['MYSQL_HOST']
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'automatch-309218-5f83b019f742.json'
CACHE_DIR = "/cache"


# preprocessing
def get_key(val, dict):
    """returns the key of the dict_units dict"""
    for key, value in dict.items():
        if val in value:
            return key


def join_units(string):
    """joins numbers with following units (ej:250 grs = 250grs)"""

    # stopwords and dict for merging units to numbers in titles
    stop_words = ['and', 'for', 'the', 'with', 'to', 'y', 'de', 'que', 'en', 'para', 'del', 'le', 'les', 'lo', 'los',
                  'la', 'las', 'con', 'que', 'gratis', 'promo', 'promocion', 'promotion', 'oferta', 'ofertas', 'free', 'gratis',
                  'descuento', 'descuentos', 'dcto', 'pagina', 'page', 'null', 'price', 'precio', 'precios', 'producto',
                  'productos', 'product', 'products', 'combo']

    # Units dict
    dict_units = {}
    dict_units['u'] = ['u', 'un', 'und' 'unit', 'units', 'unidad', 'unidades']
    dict_units['cm'] = ['cm', 'cms', 'centimeter', 'centimetros']
    dict_units['l'] = ['l', 'lt', 'lts', 'litro', 'litros', 'litre', 'litres']
    dict_units['m'] = ['m', 'mt', 'mts', 'metro', 'metros', 'meter', 'meters']
    dict_units['gr'] = ['g', 'gr', 'grs', 'gramo', 'gramos', 'gram', 'grams']
    dict_units['ml'] = ['ml', 'mls', 'mililitro', 'mililitros', 'millilitre', 'millilitres']
    dict_units['kg'] = ['kg', 'kgs', 'kilo', 'k', 'kilos', 'kilogramo', 'kilogramos', 'kilogram', 'kilograms']
    dict_units['lb'] = ['lb','lbs', 'libra', 'libras']
    dict_units['cc'] = ['cc', 'ccs']
    dict_units['mm'] = ['mm', 'mms', 'milimetro', 'milimetros', 'millimeter', 'millimeters']
    dict_units['mg'] = ['mg', 'mgs', 'miligramo', 'miligramos', 'milligram', 'milligrams']
    dict_units['gb'] = ['gb', 'gigabyte', 'gigabytes']
    dict_units['kb'] = ['kb', 'kilobyte', 'kilobytes']
    dict_units['mb'] = ['mb', 'megabyte', 'megabytes']
    dict_units['tb'] = ['tb', 'terabyte', 'terabytes']
    dict_units['w'] = ['w', 'watts']
    dict_units['hz'] = ['hz', 'hertz']
    dict_units['oz'] = ['oz', 'onza', 'onzas', 'ounce']

    dict_values = np.hstack(list(dict_units.values()))

    string_f_split = [token for token in string.split() if token not in stop_words]

    aux = []
    block_next = -1
    for index in range(len(string_f_split)):

        if index == block_next:  # skip word
            continue

        try:
            float_num = float(string_f_split[index])
            next_val = string_f_split[index + 1]
            try:
                next_float_num = float(next_val)
                if next_float_num:
                    val = string_f_split[index]
                    aux.append(val)
                    continue

            except:
                if index != len(string_f_split) and \
                        bool(re.search(r'\d', string_f_split[index + 1])) == False and \
                        string_f_split[index + 1] in dict_values:

                    val = str(float_num).rstrip('0').rstrip('.') + get_key(string_f_split[index + 1], dict_units)
                    aux.append(val)
                    block_next = index + 1
                    continue

                else:
                    val = string_f_split[index]
                    aux.append(val)
                    continue

        except:
            val = string_f_split[index]
            aux.append(val)
            continue

    formatted = ' '.join(aux)
    return formatted


def preprocess_products(row):
    """preprocess the product based on rules"""
    row = row.replace('unknown','').replace(',','.') .replace('&nbsp', ' ').replace('\xa0','').replace('"','').\
          replace("/", ' ').replace('**entrega en 2 dias habiles**','').replace('**entrega en 4 dias habiles**','').replace('**entrega en 6 dias habiles**','').\
          replace('**entrega en 7 dias habiles**','').replace('** entrega en 2 dias habiles**','').replace('** entrega en 4 dias habiles**','').\
          replace('** entrega en 6 dias habiles**','').replace('** entrega en 7 dias habiles**','').replace(' ** entrega en 4 dias habiles **','')

    row = ''.join(re.findall('[-+]?(?:\d*\.\d+|\d+)|\d|\s|\w|\+', row)).lower()
    row = row.replace('-', ' ')
    row = re.sub(r'[_]', ' ', row)
    aux = ['mas' if token == '+' else token for token in row.split()]
    row = ' '.join(aux)

    row = unidecode.unidecode(row)  # removes special characters
    row = join_units(row)  # units formatting
    row = row.split()
    row = " ".join(sorted(set(row), key=row.index))  # removes duplicates

    return row

# MySQL functions


def sku_to_data_from_sql(skus_ids_to_search: list, retail_id_to_search: int, host=MYSQL_HOST,
                         database='winodds', user='cvergara', password=MYSQL_PASSWORD):
    """extract data from mySQL based on the vector search result"""

    # formatting for the query
    if len(skus_ids_to_search) == 1:
        results_ids = tuple(skus_ids_to_search)[0]
        results_ids = f"'{results_ids}'"
        aux = '='
    elif len(skus_ids_to_search) > 1:
        results_ids = tuple(skus_ids_to_search)
        aux = 'IN'

    # Creating connection object
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    mycursor = mydb.cursor()

    query = f"SELECT wv.id AS main_web_variety_id,\
              wv.sku AS main_sku, wv.brand AS brand_to_search,\
              wv.product_name AS product_name_to_search, wv.variety_name AS variety_name_to_search,\
              wv.image AS image_to_search\
              FROM winodds.web_varieties wv\
              WHERE wv.sku {aux} {results_ids} && wv.retail_id = {retail_id_to_search} && wv.discontinued = 0"  #antes wv.discontinued =false

    mycursor.execute(query)
    results = mycursor.fetchall()

    # inserting the results into a dataframe
    cols_name = [i[0] for i in mycursor.description]
    cols_and_rows = [dict(zip(cols_name, result)) for result in results]
    df = pd.DataFrame(cols_and_rows)

    return df


def extract_from_sql(results_ids, host=MYSQL_HOST, database='winodds', user='cvergara', password=MYSQL_PASSWORD):
    """extract data from mySQL based on the vector search result"""

    # formatting for the query
    if len(results_ids) == 1:
        results_ids = tuple(results_ids)[0]
        results_ids = f"'{results_ids}'"
        aux = '='
    elif len(results_ids) > 1:
        results_ids = tuple(results_ids)
        aux = 'IN'

    # Creating connection object
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    mycursor = mydb.cursor()

    query = f"SELECT wv.id,\
              wv.brand, wv.name, wv.product_name, wv.variety_name,\
              wv.image, wv.retail_id, r.name AS retail_name_candidate, wv.url AS url_candidate, wv.sku AS sku_candidate\
              FROM winodds.web_varieties wv\
              INNER JOIN winodds.retails r ON wv.retail_id = r.id\
              WHERE wv.id {aux} {results_ids}"

    mycursor.execute(query)
    results = mycursor.fetchall()

    # inserting the results into a dataframe
    cols_name = [i[0] for i in mycursor.description]
    cols_and_rows = [dict(zip(cols_name, result)) for result in results]
    df = pd.DataFrame(cols_and_rows)

    return df


def get_available_gpus():
    """returns the available gpus for the API"""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


gpus_list = get_available_gpus()

# VIT Model
print('loading VIT image processor from HuggingFace...')
vit_preprocessing = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
print('loading VIT model')
if len(gpus_list) > 0:
    with tf.device(gpus_list[0]):
        vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR)
else:
    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR)


embedding_model_gcp = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")


def get_embedding_txt_gcp(text, model=embedding_model_gcp):
    """returns the embedding for a string using gcp embedding model"""
    text = text.replace("\n", " ")
    embedding = model.get_embeddings([text])
    embedding = embedding[0].values

    return embedding


# Image Functions

def crop_product(url_img, blur_kernel=15):
    '''crops a product from an image'''
    try:
        headers = {'User-Agent': 'Mozilla/5.0 ...'}
        # fix possible errors in url: <--- PUEDE SER MEJORADO ESTA PARTE. para casos con la url con ruido
        if url_img[:8] == 'https://':
            url_img = url_img.split('https://')[-1]
            url_img = 'https://' + url_img

        elif url_img[:7] == 'http://':
            url_img = url_img.split('http://')[-1]
            url_img = 'http://' + url_img

        elif url_img[:8] != 'https://':
            url_img = 'https:' + url_img

        response = requests.get(url_img, stream=True, headers=headers)
        img = PIL_Image.open(io.BytesIO(response.content))
        img_mode = img.mode

        # change channels to RGB
        if img_mode == 'RGBA':
            img_aux = PIL_Image.new("RGB", img.size, (255, 255, 255))
            img_aux.paste(img, mask=img.split()[3])
            img = img_aux

        if img_mode == 'CMYK':
            img = img.convert('RGB')

        if img_mode == 'P':
            img = img.convert('RGB', palette=PIL_Image.ADAPTIVE, colors=256)

        if img_mode == 'L':
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')

        img = np.array(img)
        #thresholding
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        gray = cv2.blur(gray, (blur_kernel, blur_kernel))
        thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1] #min 252 antes
        #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1] # #BUENO para el caso en que el fondo no es blanco y el color es muy distinto al del objeto
        alpha = 1  # for undefined cases : x/0 (no white pixels)
        ratio = cv2.countNonZero(thresh)/((img.shape[0] * img.shape[1]) - cv2.countNonZero(thresh) + alpha)

        if ratio > 2:  # no crop. ratio=2 good enough?
            cropped = img
            return img, cropped, thresh, ratio

        # ratio<2,  getting the max countour from img (product)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        max_a = 0
        for contour in contours:
            x_aux, y_aux, w_aux, h_aux = cv2.boundingRect(contour)
            a = w_aux * h_aux
            if a > max_a:
                max_a = a
                x, y, w, h = x_aux, y_aux, w_aux, h_aux

        cropped = img.copy()[y:y + h, x:x + w]

    except Exception as e:
        return e

    return img, cropped, thresh, ratio


def get_embedding_img(url_img: str, img_model=vit_model, crop=crop_product):
    """returns a img emebdding using the VIT model"""
    img = crop(url_img, blur_kernel=15)[1]  # cropping img
    available_gpus = get_available_gpus()
    if len(available_gpus) > 0:
        with tf.device(available_gpus[0]):
            inputs = vit_preprocessing(img, return_tensors="tf")
            embedding = img_model(**inputs).last_hidden_state[0][0].numpy()
    else:
        inputs = vit_preprocessing(img, return_tensors="tf")
        embedding = img_model(**inputs).last_hidden_state[0][0].numpy()

    return embedding.tolist()


def get_embedding_img_no_crop(url_img, img_model=vit_model):  # , crop=crop_product
    """returns a img emebdding using the VIT model"""

    headers = {'User-Agent': 'Mozilla/5.0 ...'}
    # fix possible errors in url: <--- PUEDE SER MEJORADO ESTA PARTE. para casos con la url con ruido
    if url_img[:8] == 'https://':
        url_img = url_img.split('https://')[-1]
        url_img = 'https://' + url_img

    elif url_img[:7] == 'http://':
        url_img = url_img.split('http://')[-1]
        url_img = 'http://' + url_img

    elif url_img[:8] != 'https://':
        url_img = 'https:' + url_img

    response = requests.get(url_img, stream=True, headers=headers)
    img = PIL_Image.open(io.BytesIO(response.content))

    img_mode = img.mode

    # change channels to RGB
    if img_mode == 'RGBA':
        img_aux = PIL_Image.new("RGB", img.size, (255, 255, 255))
        img_aux.paste(img, mask=img.split()[3])
        img = img_aux

    if img_mode == 'CMYK':
        img = img.convert('RGB')

    if img_mode == 'P':
        img = img.convert('RGB', palette=PIL_Image.ADAPTIVE, colors=256)

    if img_mode == 'L':
        img = img.convert('RGB')

    else:
        img = img.convert('RGB')

    img = np.array(img)
    available_gpus = get_available_gpus()
    if len(available_gpus) > 0:
        with tf.device(available_gpus[0]):
            inputs = vit_preprocessing(img, return_tensors="tf")
            embedding = img_model(**inputs).last_hidden_state[0][0].numpy()
    else:
        inputs = vit_preprocessing(img, return_tensors="tf")
        embedding = img_model(**inputs).last_hidden_state[0][0].numpy()

    return embedding.tolist()


# tools
def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def check_url(url):
    ''''returns code status from requesting a url'''
    code = requests.head(url).status_code
    return code
