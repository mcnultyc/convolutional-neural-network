#!/bin/bash

usage(){ echo "usage: ./download_images.sh -u <url file> -d <dir name> [-s]" >&2; exit 1; } 

output_dir=""
url_file=""
split_dir=0
while getopts ":u:d:hs" opt; do
  case "$opt" in
  d)  output_dir=$OPTARG
    ;;
  u)  url_file=$OPTARG
    ;;
  s)  split_dir=1
    ;;
  h)  usage
    ;;
    esac
done

if [ -z "$url_file" ]; then
  echo "error: -u <url file>, required argument" >&2
  exit 1  
fi

if [ -z "$output_dir" ]; then
  echo "error: -d <dir name>, required argument" >&2
  exit 1
fi

mkdir $output_dir
cd $output_dir

if [ $split_dir -eq 1 ]; then
  mkdir "test"
  mkdir "train"
fi

i=0
while read line
do  
  wget $line -q -T 20 -t 4 -O "image_$i.jpg"
  wget_ret="$?"
  if [ $wget_ret -eq 0 ]; then
    file_output=$(file -b --mime-type "image_$i.jpg")
    file_ret="$?"
    if [ $file_ret -eq 0 ] && [ "$file_output" = "image/jpeg" ]; then
      i=$((i + 1))
    else
      rm "image_$i.jpg"
    fi  
  else
    rm "image_$i.jpg"
  fi
done < "../$url_file"

if [ $split_dir -eq 1 ]; then 
  test_size=$(echo $i | awk '{printf "%d\n",$1*0.2}')
  train_size=$(( i - $test_size ))

  i=0
  for image in *.jpg; do
    if [ $i -ge $test_size ]; then
      mv $image "train" 
    else
      mv $image "test"
    fi
    i=$(( i + 1 ))
  done
fi
