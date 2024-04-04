#!/usr/bin/env python
# -*- coding: utf-8 -*-

def lambda_handler(event, context):
    json = {"statusCode": 200, "body": "hello world"}
    return json
