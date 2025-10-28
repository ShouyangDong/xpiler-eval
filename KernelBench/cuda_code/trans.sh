
#!bin/sh
for file in ./*
do
    hipify-perl $file > $file".hip"
    echo $file".hip"
done
