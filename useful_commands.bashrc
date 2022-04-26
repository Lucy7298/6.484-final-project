MODEL_LOG_DIR='/nobackup/users/yunxingl/models'

make_exp_dir () {
    dirpath=$1
    #logpath="$TRAINING_LOG_DIR/$__dirname"

    if [ -d "$dirpath" ]; then
    ### Take action if $DIR exists ###
    echo "Error: ${dirpath} already exists. Can not continue."
    exit 1
    else
    ###  Control will jump here if $DIR does NOT exists ###
    echo "Making directory ${dirpath}..."
    fi

    mkdir $dirpath
    mkdir $dirpath/log
    mkdir $dirpath/checkpoint
}