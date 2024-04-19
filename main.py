# libraries
from utils import *
import pandas as pd
from modal import Image, Secret, Mount, Stub, asgi_app, gpu, NetworkFileSystem
from fastapi import FastAPI, Request
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Annotated
import os
from io import BytesIO
import json

q_gpu = 1
GPU = gpu.A10G(count=q_gpu)
volume = NetworkFileSystem.persisted("model-cache-vol-3")
CACHE_DIR = "/cache"
stub = Stub(name="api-generate-embeddings")

conda_image = (Image.conda()
               .conda_install(
                "cudatoolkit=11.2",
                "cudnn=8.1.0",
                "cuda-nvcc",
                channels=["conda-forge", "nvidia"],
                )
               .pip_install("pandas", "numpy", "matplotlib", "requests", "Pillow", "opencv-python-headless",
                            "jax", "jaxlib", "transformers==4.29.2", "tensorflow~=2.9.1",
                            "Unidecode", "python-dotenv", "mysql-connector-python",
                            "scikit-learn", "google-cloud-aiplatform==1.25", "python-jose[cryptography]",
                            "passlib[bcrypt]", "python-multipart", "openpyxl"))


class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers['X-Process-Time'] = str(process_time)
        return response


embeddings_app = FastAPI(title='EmbeddingsGeneratorAPI',
                       summary="Embeddings: txt & img generator API - Geti", version="1.0",
                       contact={
                                "name": "Cristian Vergara",
                                "email": "cvergara@geti.cl",
                                })
embeddings_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'])
embeddings_app.add_middleware(MyMiddleware)


@embeddings_app.get("/")
def read_root():
    return {"Root": "Root_test"}


@embeddings_app.get("/txt-gcp-embedding")
async def return_embedding(sku_to_search: str, retail_id_to_search: int):
    try:
        sku_to_search = sku_to_search.strip()
        df_product = sku_to_data_from_sql(skus_ids_to_search=[sku_to_search], retail_id_to_search=retail_id_to_search)
        df_product.drop_duplicates(inplace=True)
        print(df_product)
        df_product['brand_and_product'] = df_product.brand_to_search + ' ' + df_product.product_name_to_search + ' ' + df_product.variety_name_to_search
        df_product['brand_and_product'] = df_product['brand_and_product'].apply(lambda row: preprocess_products(str(row)))
        brand_and_product = df_product.brand_and_product.tolist()[0]
        txt_embedding = get_embedding_txt_gcp(brand_and_product, model=embedding_model_gcp)

        output = {"sku": str(sku_to_search),
                  "retail_id": int(retail_id_to_search),
                  "brand_and_product": str(brand_and_product),
                  "txt_embedding": txt_embedding}

        return output

    except Exception as e:
        output = {
            "sku": str(sku_to_search),
            "retail_id": int(retail_id_to_search),
            "error": str(e),
            "txt_embedding": []
        }
        return output


@embeddings_app.get("/img-vit-embedding")
async def return_img_embedding(sku_to_search: str, retail_id_to_search: int):
    try:
        sku_to_search = sku_to_search.strip()
        df_product = sku_to_data_from_sql(skus_ids_to_search=[sku_to_search], retail_id_to_search=retail_id_to_search)
        df_product.drop_duplicates(inplace=True)
        print(df_product)
        df_product['brand_and_product'] = df_product.brand_to_search + ' ' + df_product.product_name_to_search + ' ' + df_product.variety_name_to_search
        df_product['brand_and_product'] = df_product['brand_and_product'].apply(lambda row: preprocess_products(str(row)))
        img_url = df_product.image_to_search.tolist()[0]
        brand_and_product = df_product.brand_and_product.tolist()[0]
        try:
            check_url(img_url)
        except Exception as e:
            response = {
                "error": str(e),
                "url with error": str(img_url),
            }
            return response

        img_embedding = get_embedding_img_no_crop(img_url, img_model=vit_model)

        output = {"sku": str(sku_to_search),
                  "retail_id": int(retail_id_to_search),
                  "brand_and_product": str(brand_and_product),
                  "img_url": str(img_url),
                  "img_embedding": img_embedding}

        return output

    except Exception as e:
        output = {
            "sku": str(sku_to_search),
            "retail_id": int(retail_id_to_search),
            "error": str(e),
            "img_embedding": []
        }
        return output


@stub.function(image=conda_image, gpu=GPU,      #keep_warm=1,
               secret=Secret.from_name("automatch-secret-keys"),
               mounts=[Mount.from_local_file("automatch-309218-5f83b019f742.json",
                       remote_path="/root/automatch-309218-5f83b019f742.json")],
               network_file_systems={CACHE_DIR: volume},
               timeout=999)  # schedule=Period(minutes=30)
@asgi_app(label='generate-embeddings-geti')
def fastapi_embeddings_app():
    # check available GPUs
    print(get_available_gpus())

    return embeddings_app

