#%%
import torch as t
import patch_tranformer_lens
import sys
sys.path.append("/data/shenth/work/auto-circuit")
import auto_circuit
from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import draw_seq_graph, GraphManager

# device = t.device("cuda" if t.cuda.is_available() else "cpu")
device = t.device("cpu")
model = load_tl_model("/data/shenth/models/deepseek/v2-lite", device)

# path = repo_path_to_abs_path("datasets/ioi/ioi_vanilla_template_prompts.json")
# train_loader, test_loader = load_datasets_from_json(
#     model=model,
#     path=path,
#     device=device,
#     prepend_bos=True,
#     batch_size=16,
#     train_test_size=(128, 128),
# )

# model = patchable_model(
#     model,
#     factorized=True,
#     slice_output="last_seq",
#     separate_qkv=True,
#     device=device,
# )

# attribution_scores: PruneScores = mask_gradient_prune_scores(
#     model=model,
#     dataloader=train_loader,
#     official_edges=None,
#     grad_function="logit",
#     answer_function="avg_diff",
#     mask_val=0.0,
# )

# THRESHOLD = 3.5
# fig, gm = draw_seq_graph(
#     model, attribution_scores, THRESHOLD, layer_spacing=True, orientation="v", draw_fig=False
# )
# if fig is not None:
#     fig.write_image(repo_path_to_abs_path("docs/assets/IOI_Attributions_Viz.png"), scale=4)
# nodes = gm.get_nodes()
# print(f"Threshold: {THRESHOLD}, Nodes: {len(nodes)}")