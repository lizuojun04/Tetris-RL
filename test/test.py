"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import torch
import cv2
from src.tetris import Tetris
from src.agents.model import TestModel, TetrisRL
from src.agents.heuristic_agent import HeuristicAgent


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="checkpoints")
    parser.add_argument("--output_path", type=str, default="output_video")
    parser.add_argument("--agent", type=str, default="base",
                        choices=["base", "heuristic"],
                        help="Choose the agent: base, heuristic")
    parser.add_argument("--test", type=int, default=10, help="test times")
    parser.add_argument("--render", action="store_true", help="Enable rendering to video.")

    args = parser.parse_args()
    return args


def test(opt):
    if not os.path.exists(opt.output_path):
        os.mkdir(opt.output_path)

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
    if opt.agent == "heuristic":
        agent = HeuristicAgent()
    else:
        agent = TetrisRL()
        if torch.cuda.is_available():
            agent_dict = torch.load("./checkpoints/DQN_best.pt", weights_only=False)
        else:
            agent_dict = torch.load("./checkpoints/DQN_best.pt", map_location=lambda storage, loc: storage, weights_only=False)
        agent.load_state_dict(agent_dict)
        agent.eval()
        if torch.cuda.is_available():
            agent.cuda()

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    test_times = opt.test
    total_score = 0
    for i in range(test_times):
        if opt.render:
            video_name = os.path.join(opt.output_path, f"game_{i}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
            out = cv2.VideoWriter(video_name, fourcc, opt.fps,
                                  (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
        else:
            out = None
        env.reset()
        while True:
            next_steps = env.get_next_states()
            action = agent.get_action(next_steps)
            _, done = env.step(action, render=opt.render, video=out)

            if done:
                if out is not None:
                    out.release()
                break
        print(env.score)
        total_score += env.score

    print(total_score/test_times)
        


if __name__ == "__main__":
    opt = get_args()
    test(opt)
