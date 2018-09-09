#!/bin/bash

cd /home/bill/datadisk/runavaweb/src/
source /home/bill/anaconda3/bin/activate avarunenv
python site_update_dev.py
source /home/bill/anaconda3/bin/deactivate
cd /home/bill

