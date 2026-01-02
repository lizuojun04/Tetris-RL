import argparse
import torch
import cv2
from src.tetris import Tetris

class HeuristicAgent:
    def __init__(self):
        self.weights = {
            "lines": 0.76,
            "holes": -0.36,
            "bumpiness": -0.18,
            "total_height": -0.51,
            "max_height": 0
        }

    def get_action(self, next_steps):
        """
        next_steps: env.get_next_states() 返回的字典
        Values 是 (grid, features)
        features = [lines, holes, bumpiness, height]
        """
        best_score = -float('inf')
        best_action = None

        for action, (grid, feats) in next_steps.items():
            lines = feats[0].item()
            holes = feats[1].item()
            bumpiness = feats[2].item()
            total_height = feats[3].item()
            max_height = feats[4].item()

            score = (self.weights["lines"] * lines +
                     self.weights["holes"] * holes +
                     self.weights["bumpiness"] * bumpiness +
                     self.weights["total_height"] * total_height +
                     self.weights["max_height"] * max_height)

            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
