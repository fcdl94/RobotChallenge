#!/bin/bash
python -m visdom.server > vis.log 2> vis.err &
sleep 5
python vis.py
