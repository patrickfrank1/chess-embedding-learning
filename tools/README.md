# Tools that help dealing with large pgn files

## To unzip pgn files

pbzip2 is much faster than bzip2. Other tools can be used too.

    $ pbzip2 -d *.pgn.bz2

## To split pgn files into smaller files

Use https://github.com/cyanfish/pgnsplit rather than 'split', it respects game boundaries and produces valid pgn files.
To use in on linux, install mono, then run:
    
        $ sudo apt-get install mono-complete
        $ mono PgnSplit.exe

## To extract chess positions from pgn files

I provide a cli tool. It can easily be configured to extract positions to various formats including bitboards and tensorflow tensors. A template config fie can be found in src/preprocessing/config.yaml.

    $ ./extract_positions.sh

This preprocessing step is used to generate the training data for the neural network.
