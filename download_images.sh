#!/bin/bash

output_dir=""
url_file=""
while getopts "f:u:" opt; do
	case "$opt" in
	f)  output_dir=$OPTARG
		;;
	u)	url_file=$OPTARG
		;;
    esac
done

if [ -z "$url_file" ]; then
	echo "Error: url file missing"
	exit 1	
fi

if [ -n "$output_dir" ]; then
	mkdir $output_dir
	cd $output_dir
fi

i=0
while read line
do	
	wget $line -q -t 4 -O "image_$i.jpg"
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
