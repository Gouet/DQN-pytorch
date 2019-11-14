call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL

python train.py --load-episode-saved 2000 --eval
pause