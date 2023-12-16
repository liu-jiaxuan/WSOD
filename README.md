## 1. Requirements

Python 3.7+, Pytorch 1.9.0, Cuda 11.1, TensorboardX 2.4, opencv-python

For detailed environment configuration, please refer to "./environment/env.txt" (for pip) and  "./environment/env.yaml" (for conda).


## 2. Training & Testing

- Train the model:

    `bash train_best_model.sh `

- Test the model:

    `bash  test_model.sh`
    
    The predicted saliency maps will be saved in "./Our_Proposed_Best_Model/pred_maps".
    The metrics results will be saved in "./Our_Proposed_Best_Model/score/result.txt"


## 3. single-bounding-box annotations:
Our proposed annotations for the trainset can be downloaded [here](https://drive.google.com/file/d/1qWFoc8zTbomdXPTl2KX9ablXXFOciD1W/view?usp=sharing).
