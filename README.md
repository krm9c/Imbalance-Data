# Policy Gradient 


# ---------------- Requirement ----------------

python==3.7.10

pytorch==1.7.1

torchvision==0.8.2

gym==0.21.0

numpy==1.21.5

pandas==1.2.4

matplotlib==3.5.0

tqdm==4.59.0

importlib-metadata==4.13.0

pyglet==1.5.21

# ---------------- Usage ----------------
## Policy Gradient 
Please run
```
python3 main_pg.py --scratch True --task "CartPole-v1" --max_steps 100 --optimizer "Adam" --num_episodes 10000 --learn_rate 1e-3 --save_after 1000
```



