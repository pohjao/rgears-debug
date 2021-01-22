# rgears-debug
0. `pip install -r pip_requirements.txt`
1. `git lfs install`
2. `git lfs fetch --all`
3. `docker build -t test-redis . && docker run -it --rm -p 6379:6379 test-redis`
4. `python run.py`
5. Crash should occur

If you run `python run_redisai_client.py` no problems occurs. The script uses redisai-client to run operations.

The model is a frozen Tensorflow graph. It is converted from Keras model hosted [here](https://github.com/priya-dwivedi/aerial_pedestrian_detection). Model input and output details are in `gear.py`