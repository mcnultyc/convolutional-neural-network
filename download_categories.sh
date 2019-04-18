#!/bin/bash

usage(){ echo "usage: ./download_categories.sh -d <directory> -f <filename>" >&2; exit 1; } 

download_files(){  
  local i=0
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
  done < "../../$1"
  echo $i
}

split_files(){
  local test_size=$(echo $2 | awk '{printf "%d\n",$1*0.2}')
  local train_size=$(( $2 - $test_size ))

  local i=0
  for image in *.jpg; do
    if [ $i -ge $test_size ]; then
      mv $image "../train/$1" 
    else
      mv $image "../test/$1"
    fi
    i=$(( i + 1 ))
  done
}

directory=""
categories_file=""
while getopts ":d:f:h" opt; do
  case "$opt" in
  d)  directory=$OPTARG
    ;;
  f)  categories_file=$OPTARG
    ;;
  h)  usage
    ;;
    esac
done

if [ -z "$directory" ] || [ -z "$categories_file" ]; then
  usage 
fi

mkdir $directory
cd $directory

mkdir "test"
mkdir "train"

i=0
while read line
do
  tokens=($line)
  url_file=${tokens[0]}
  category=${tokens[1]}
  mkdir $category
  mkdir "test/$category"
  mkdir "train/$category"
  cd $category
  i=$(download_files $url_file)
  split_files $category $i
  cd ..
  rm -r $category
done < "../$categories_file"
