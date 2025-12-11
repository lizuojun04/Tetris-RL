"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
from src.tetris import Tetris
from src.agents.model import TestModel, TetrisRL


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    """
    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path), weights_only=False)
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage, weights_only=False)
    """
    # model = TestModel()
    model = TetrisRL()
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions = list(next_steps.keys())
        # stack for batch dimension
        # (states_num, 1, 20, 10)
        next_grids = torch.stack([v[0] for v in next_steps.values()])
        # (status_num, 4)
        next_feats = torch.stack([v[1] for v in next_steps.values()])
        if torch.cuda.is_available():
            next_grids = next_grids.cuda()
            next_feats = next_feats.cuda()
        predictions = model(next_grids, next_feats)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break

    print(env.score)
        


if __name__ == "__main__":
    opt = get_args()
    test(opt)
