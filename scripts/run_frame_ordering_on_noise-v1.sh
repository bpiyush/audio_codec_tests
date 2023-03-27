python train_frame_ordering.py --only_train --gpus 0 && \
python train_frame_ordering.py --only_train --gpus 1 --codec raw --audio_fps 16000 && \
python train_frame_ordering.py --only_train --gpus 2 --codec aac && \
python train_frame_ordering.py --only_train --gpus 3 --codec pcms16le
