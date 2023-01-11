import argparse
import matplotlib.pyplot as plt
import numpy as np

def main(configs):
  """
  ==========
  plot(x, y)
  ==========

  See `~matplotlib.axes.Axes.plot`.
  """

  cumulative_auc = np.load(configs['npy_path'])
  th = configs['th']
  res = configs['res']
  # plt.style.use('_mpl-gallery')
  #print('linspace',linspace)
  print('cumulative auc',cumulative_auc)
  # make data
  x = np.linspace(0.1*th, th, num=res)
  y = cumulative_auc

  # plot
  fig, ax = plt.subplots()

  ax.plot(x, y, linewidth=2.0)

  ax.set(xlim=(0, th+.0005),
        ylim=(0, 1+.005))

  plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AUC_graph')
    parser.add_argument('--npy', type=str) # config file
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--threshold', type=float)
    # parser.add_argument('--ignore_bit', type=str)
    # parser.add_argument('--eval_output_path', type=str)
    # parser.add_argument('--use_icp', type=str, choices=('True','False'), default='False') # config file
    args = parser.parse_args()
    # config_file = args.cfg
    # checkpoint_file = args.ckpt_file
    # print('arg types \nnpy path', type(args.npy),type(args.resolution),type(args.threshold))
    npy_path = args.npy
    resolution = args.resolution
    threshold = args.threshold
    # print('types \nnpy path', type(npy_path),'resolution',type(resolution),'threshold',type(threshold))
    configs = {
        'res': resolution,
        'npy_path': npy_path,
        'th': threshold,

    # configs['res'] = resolution
    # configs['npy_path'] = npy_path
    # configs['th'] = threshold
    }
    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)
