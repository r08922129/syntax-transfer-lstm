for level in {2..8};
do
    python main.py --train --batch_size=2 --save_path=save/level$level --decode_level=$level;

done
