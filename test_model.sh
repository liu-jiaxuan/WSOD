"""
ICCV 2023
Paper ID: 6886
"""

# Predicted saliency maps are saved in "./Our_Proposed_Best_Model/pred_maps"
# Metrics results are saved in "./Our_Proposed_Best_Model/score/result.txt"

tag = 'Our_Proposed_Best_Model'

python test.py  --tag ${tag} --save True
cd evaluation
python main.py --base ../${tag}