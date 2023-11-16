from llff.poses.pose_utils import gen_poses
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--match_type', type=str,  # 这里指匹配的种类。一种是穷举匹配器，一种是顺序匹配器
					default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
parser.add_argument('--scenedir', type=str, # 这里填写的是做过colmap处理的那个项目路径
					default="E:/COLMAP-3.7-windows-cuda/COLMAP-3.7-windows-cuda/project3",
                    help='input scene directory')
args = parser.parse_args()

if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
	print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
	sys.exit()

if __name__=='__main__':
    gen_poses(args.scenedir, args.match_type)