#!/bin/bash


while true; do 

    hadoop fs -put -f $1 $2

    sleep 300

done