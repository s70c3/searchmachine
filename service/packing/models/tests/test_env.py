import requests as r
import random
# import yaml


def send_post_request(addr, json_params):
    resp = r.post(addr, json=json_params)
    return resp.json()