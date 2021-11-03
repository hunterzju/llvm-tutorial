for _file in `dir $PWD`
do
    if [ "${_file##*.}" = "rst" ];
    then
        pandoc ${_file} -f rst -t markdown -o ../markdown/${_file/rst.txt/md}
    fi
done
