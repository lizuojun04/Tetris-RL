## 依赖项

---

这里的 `https://download.pytorch.org/whl/cu124` 需要根据自己的显卡驱动版本更改
```sh
conda create -n tetris python=3.10
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```


## 使用

---

```sh
cd Tetris-RL
python -m test.test
```

## Agent

---

### Heuristic

---

启发式算法，遍历可能采取的所有动作，然后查看棋盘状态，选择使得棋盘状态最好的动作

关于棋盘状态怎么样算最好，这取决于人类的定义，也就是说启发式算法不会进行学习，
他有多强取决于人类将规则定义得多好
