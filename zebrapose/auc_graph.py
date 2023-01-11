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
  id=configs['obj_id']
  th = configs['th']
  res = configs['res']

  fig, ax = plt.subplots(3)

  ax[0].set_title('Cumulative ADD score for multiple thresholds for object {}'.format(id))
  
  
  for i in range(0,3):
    # plt.style.use('_mpl-gallery')
    #print('linspace',linspace)
    print('cumulative auc',cumulative_auc[i])
    # make data
    x = np.linspace(0.1*th[i], th[i], num=res)
    y = cumulative_auc[i]

    # plot

    ax[i].plot(x, y, linewidth=2.0)

    ax[i].set(xlim=(0, th[i]+.0005),
            ylim=(0, 1+.005))
    #ax[i].set_title('Cumulative ADD_{}% '.format(th[i]*100))
  plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AUC_graph')
    parser.add_argument('--npy', type=str) # numpy file containing the array with the cumulative ADD data
    parser.add_argument('--obj_id', type=int)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()
    
    npy_path = args.npy
    obj_id = args.obj_id
    resolution = args.resolution
    threshold = args.threshold
    
    configs = {
        'res': resolution,
        'obj_id': obj_id,
        'npy_path': npy_path,
        'th': [threshold,0.5*threshold, 0.25*threshold],
    }
    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)
