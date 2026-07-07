import os
import json
from datetime import datetime
from collections import defaultdict
from itertools import chain

from GOOD import config_summoner
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.logger import load_logger
from GOOD.utils.metric import assign_dict
from GOOD.utils.loader import initialize_model_dataset
from GOOD.networks.models.DIRGNN import split_graph_node
import GOOD.kernel.pipelines.xai_metric_utils as xai_utils
from GOOD.definitions import ROOT_DIR

import numpy as np
import torch

from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max, scatter_add

import matplotlib.pyplot as plt

def gmean(a,b):
    return (a*b).sqrt()

def aritm(a,b):
    return (a+b) / 2

def armonic(a,b):
    return 2 * (a*b) / (a+b)

def gstd(a):
    return a.log().std().exp()

def print_metric(name, data, results_aggregated=None, key=None):
    avg = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    print(f"{name:<25}", " = ", ", ".join([f"{avg[i]:.3f} +- {std[i]:.3f}" for i in range(len(avg))]))
    if not results_aggregated is None:
        assign_dict(
            results_aggregated,
            key,
            avg.tolist()
        )
        key[-1] += "_std" # add _std to the metric name
        assign_dict(
            results_aggregated,
            key,
            std.tolist()
        )

def get_tensorized_metric(scores, c):
    return torch.tensor([
        scores[i][c] for i in range(len(scores))
    ])

def stability_detector_rebuttal(args):
    assert len(args.metrics.split("/")) == 1, args.metrics.split("/")
    
    load_splits = ["id"]
    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val"]
    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    # ratios = [0.3]
    print("Using ratios = ", ratios)

    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"COMPUTING STABILITY DETECTOR FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            # if "CIGA" in config.model.model_name:
            #     ratios = [pipeline.model.att_net.ratio]
            
            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
            )
            intervention_bank = None

            for split in splits:
                score, acc_int, _ = pipeline.compute_stability_detector_rebuttal(
                    seed,
                    ratios,
                    split,
                    metric=args.metrics,
                    intervention_distrib=config.intervention_distrib,
                    intervention_bank=intervention_bank,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    graphs_nx=graphs_nx[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                    causal_subgraphs_r=causal_subgraphs_r[split],
                    spu_subgraphs_r=spu_subgraphs_r[split],
                    expl_accs_r=expl_accs_r[split],
                    causal_masks_r=causal_masks_r[split]
                )
                results[load_split][split][args.metrics].append(score["all_L1"])
                for m in ["wiou_original", "wiou_perturbed"]:
                    for s in [""]:
                        results[load_split][split][m + s].append(score[m + s])
    
    print(f"\n\nDONE {config.dataset.dataset_name} - {config.model.model_name}")
    for load_split in results.keys():
        for split in results[load_split]:
            for metric in ["wiou_original", "wiou_perturbed"]:
                matrix = np.array(results[load_split][split][metric])
                mean_over_seeds = np.mean(matrix, axis=0)
                mean_over_ratios = np.mean(mean_over_seeds)
                std = np.std(matrix, axis=0)
                print(f"({load_split}) {split} {metric.upper()}: \t{mean_over_seeds} +- {std} (mean across ratios: {mean_over_ratios})")

def stability_detector_extended(args):
    startOverallTime = datetime.now()
    assert len(args.metrics.split("/")) == 1, args.metrics.split("/")
    
    load_splits = ["id"]
    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val"]

    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    print("Using ratios = ", ratios)

    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"COMPUTING STABILITY DETECTOR FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 
            
            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
            )
            intervention_bank = None

            for split in splits:
                score, acc_int, _ = pipeline.compute_stability_detector_extended(
                    seed,
                    ratios,
                    split,
                    metric=args.metrics,
                    intervention_distrib=config.intervention_distrib,
                    intervention_bank=intervention_bank,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    graphs_nx=graphs_nx[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                    causal_subgraphs_r=causal_subgraphs_r[split],
                    spu_subgraphs_r=spu_subgraphs_r[split],
                    expl_accs_r=expl_accs_r[split],
                    causal_masks_r=causal_masks_r[split]
                )
                results[load_split][split][args.metrics].append(score["all_L1"])
                for m in ["plausibility_wiou", "stability_wiou", "stability_f1", "stability_mcc"]:
                    for s in ["_original", "_perturbed"]:
                        results[load_split][split][m + s].append(score[m + s])
    
    print(f"\n\FINISHED {config.dataset.dataset_name} - {config.model.model_name}")
    for load_split in results.keys():
        for split in results[load_split]:
            for m in ["plausibility_wiou", "stability_wiou", "stability_f1", "stability_mcc"]:
                for s in ["_original", "_perturbed"]:
                    metric = m + s
                    matrix = np.array(results[load_split][split][metric])
                    mean_over_seeds = np.mean(matrix, axis=0)
                    mean_over_ratios = np.mean(mean_over_seeds)
                    std_over_seeds = np.std(matrix, axis=0)
                    std_over_ratios = np.sqrt(1/(mean_over_seeds.shape[0]**2) * np.sum(std_over_seeds**2)) # Var((a+b)/2) = 1/4[Var(a) + Var(b) - Cov(a,b)] (assuming a indip. b Cov(a,b)=0)
                    print(f"({load_split}) ({split}) {metric.upper()}: \t{mean_over_seeds} +- {std_over_seeds} (mean across ratios: {mean_over_ratios:.2f} +- {std_over_ratios:.2f})")
                print()
    print("Overall time of execution: ", datetime.now() - startOverallTime)

def plot_explanation_examples(args):
    assert len(args.metrics.split("/")) == 1, args.metrics.split("/")
    
    load_splits = ["id"]
    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val"]
    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    # ratios = [0.3]
    print("Using ratios = ", ratios)

    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"PLOTTING EXAMPLES OF EXPLANATIONS FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 
            
            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
            )
            intervention_bank = None

            for split in splits:
                pipeline.generate_explanation_examples(
                    seed,
                    ratios,
                    split,
                    metric=args.metrics,
                    intervention_distrib=config.intervention_distrib,
                    intervention_bank=intervention_bank,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    graphs_nx=graphs_nx[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                    causal_subgraphs_r=causal_subgraphs_r[split],
                    spu_subgraphs_r=spu_subgraphs_r[split],
                    expl_accs_r=expl_accs_r[split],
                    causal_masks_r=causal_masks_r[split]
                )

def permute_attention_scores(args):
    load_splits = ["id"]
    splits = ["id_val", "val", "test"]
    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING ATTN. PERMUTATION FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            for s in splits:
                acc_ori, acc = pipeline.permute_attention_scores(s)
                results[load_split][s]["ori"].append(acc_ori)
                results[load_split][s]["perm"].append(acc)
    
    print(f"{config.dataset.dataset_name} - {config.model.model_name}")
    for load_split in results.keys():
        for split in results[load_split]:
            print(f"({load_split}) {split} {loader[split].dataset.metric} orig.: \t{np.mean(results[load_split][split]['ori']):.3f} +- {np.std(results[load_split][split]['ori']):.3f}")
            print(f"({load_split}) {split} {loader[split].dataset.metric} perm.: \t{np.mean(results[load_split][split]['perm']):.3f} +- {np.std(results[load_split][split]['perm']):.3f}")

def generate_panel(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        edge_scores_seed = []
        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING PLOT FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            edge_scores = pipeline.generate_panel()
            edge_scores_seed.append(edge_scores)
        pipeline.generate_panel_all_seeds(edge_scores_seed)

def generate_plot_sampling(args):
    load_splits = ["id"]
    splits = ["test"]
    seeds = args.seeds.split("/")
    ratios = [0.3, 0.6, 0.8, 0.9, 1.0] #[0.3, 0.45, 0.6, 0.75, 0.9]   0.3, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95
    sampling_alphas = [0.03, 0.05]
    all_metrics, all_accs = {}, {}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(seeds):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)

            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx="suff" in args.metrics
            )
            
            metrics, accs = pipeline.generate_plot_sampling_type(splits, ratios, sampling_alphas, graphs, graphs_nx, causal_subgraphs_r, causal_masks_r, avg_graph_size)
            all_metrics[str(seed)] = metrics
            all_accs[str(seed)] = accs

            # print(all_metrics.keys())
            # print(all_metrics["1"].keys())
            # print(all_metrics["1"]["test"].keys())
            # print(all_metrics["1"]["test"][0.3].keys())
            # print(all_metrics["1"]["test"][0.3]["RFID_0.03"])
        
        for SPLIT in splits:
            num_cols = len(sampling_alphas)
            fig, axs = plt.subplots(1, num_cols, figsize=(2.9*num_cols, 3.9), sharey=True)
            colors = {
                "NEC KL": "blue", "NEC L1":"lightblue", "FID L1 div": "green", "Model FID": "orange", "Phen. FID": "red", "Change pred": "violet"
            }
            sampling_name = {"RFID_": "RFID+ ($)", "FIXED_": "Fixed Deconfounded ($)", "DECONF_": "NEC ($)", "DECONF_R_": "NEC ($)"}
            for j, sampling_type_ori in enumerate(["RFID_", "DECONF_", "DECONF_R_"]): #"FIXED_", 
                for alpha_i, alpha in enumerate(sampling_alphas):
                    param = str(alpha_i+1 if sampling_type_ori == "FIXED_" else alpha)
                    sampling_type = sampling_type_ori + param
                    anneal, anneal_std = [], []
                    for r in ratios:
                        for i, metric_name in enumerate(["NEC L1"]):
                            anneal.append(np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]))
                            anneal_std.append(np.std([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]))
                            # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]), label=f"{metric_name}" if r == 0.3 else None, c=colors[metric_name])
                            # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]) - np.std([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]), c=colors[metric_name], alpha=.5, marker="^")
                            # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]) + np.std([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]), c=colors[metric_name], alpha=.5, marker="v")
                        # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_accs[s][SPLIT][r] for s in seeds]), label=f"Model acc" if r == 0.3 else None, c="orange", alpha=0.5)
                        # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_accs[s][SPLIT][r] for s in seeds]) - np.std([all_accs[s][SPLIT][r] for s in seeds]), c="orange", alpha=.5, marker="^")
                        # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_accs[s][SPLIT][r] for s in seeds]) + np.std([all_accs[s][SPLIT][r] for s in seeds]), c="orange", alpha=.5, marker="v")
                    
                    if "RFID" in sampling_type:
                        l = f"$\kappa=${param}"
                    elif "DECONF_R_" in sampling_type:
                        l = f"$b=${param}||R||"
                    elif "DECONF_" in sampling_type:
                        l = f"$b=${param}" + "$\\bar{m}$"

                    axs[alpha_i%num_cols].errorbar(
                        ratios,
                        anneal,
                        yerr=anneal_std,
                        fmt='-o',
                        capsize=5,
                        label=sampling_name[sampling_type_ori].replace('$', l))
                    # axs[j%num_cols,alpha_i%num_cols].plot(ratios, anneal)
                    # axs[alpha_i%num_cols].plot(ratios, [np.mean([all_accs[s][SPLIT][r] for s in seeds]) for r in ratios], c="orange", alpha=0.5)
                    axs[alpha_i%num_cols].grid(visible=True, alpha=0.5)
                    # axs[alpha_i%num_cols].set_title(f"{sampling_name[sampling_type_ori].replace('$', str(param))}")
                    # axs[alpha_i%num_cols].set_xlabel("ratio")
                    # axs[alpha_i%num_cols].set_ylabel("metric value")
                    axs[alpha_i%num_cols].set_ylim((0., 1.1))
                    axs[alpha_i%num_cols].legend(loc='best', fontsize=11)
            # plt.suptitle(f"{config.dataset.dataset_name}/{config.dataset.domain}")
            fig.supxlabel('size ratio', fontsize=13)
            fig.supylabel('value', fontsize=13)
            
            # plt.xticks(fontsize=12)
            plt.legend()
            # fig.subplots_adjust(bottom=0.3, top=0.95, left=0.1, right=0.9)
            plt.tight_layout()
            plt.savefig(f"./GOOD/kernel/pipelines/plots/metrics/R_dev_nec_sampling_{config.ood.ood_alg}_{config.dataset.dataset_name}_{config.dataset.domain}_({SPLIT}).png")
            # plt.savefig(f"./GOOD/kernel/pipelines/plots/metrics/pdfs/small_v2_dev_nec_sampling_{config.ood.ood_alg}_{config.dataset.dataset_name}_{config.dataset.domain}_({SPLIT}).pdf")
            plt.show(); 

def evaluate_metric_reconsidering(args):
    load_splits = ["id"]

    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val", "val", "test"] #"id_val", "val", "test"
    print("Using splits = ", splits)
        
    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    print("Using ratios = ", ratios)
    startTime = datetime.now()

    metrics_score = {}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")
        # with open(f"storage/metric_results/{load_split}_results.json", "r") as jsonFile:
        #     results_big = json.load(jsonFile)

        metrics_score[load_split] = {s: defaultdict(list) for s in splits + ["test", "test_R"]}
        for i, seed in enumerate(args.seeds.split("/")):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            expname = f"{load_split}_{config.util_model_dirname}_{config.dataset.dataset_name}{config.dataset.domain}" \
                f"_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}" \
                f"_samplingtype{config.samplingtype}_necnumbersamples{config.nec_number_samples}"\
                f"_nec_alpha_1{config.nec_alpha_1}_fidelity_alpha_2{config.fidelity_alpha_2}"
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)
            if "CIGA" in config.model.model_name:
                ratios = [pipeline.model.att_net.ratio, 1.0]

            if not (len(args.metrics.split("/")) == 1 and args.metrics.split("/")[0] == "acc"):
                (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
                causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                    ratios,
                    splits,
                    convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics),
                    is_weight="weight" in config.log_id
                )
                intervention_bank = None
                # if "suff" in args.metrics:
                #     intervention_bank = pipeline.compute_intervention_bank(ratios, splits=["id_val", "val", "test"], graphs_nx=graphs_nx, causal_subgraphs_r=causal_subgraphs_r)

            for metric in args.metrics.split("/"):
                print(f"\n\nEvaluating {metric.upper()} for seed {seed} with load_split {load_split}\n")

                if metric == "acc":
                    assert not (config.acc_givenR and config.mask)
                    if not config.acc_givenR:
                        for split in splits + (["test"] if not "test" in splits else []):
                            pipeline.compute_accuracy_binarizing(split, givenR=False, metric_collector=metrics_score[load_split][split])
                    print("\n\nComputing now with givenR...\n")
                    pipeline.compute_accuracy_binarizing("test", givenR=True, metric_collector=metrics_score[load_split]["test_R"])
                    continue
                elif metric == "plaus":
                    for split in splits:
                        metrics_score[load_split][split]["wiou"].append([np.mean([e[0] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["wiou_std"].append([np.std([e[0] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["F1"].append([np.mean([e[1] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["F1_std"].append([np.std([e[1] for e in expl_accs_r[split][r]]) for r in ratios])
                    continue

                for split in splits:
                    score, acc_int, results = pipeline.compute_metric_ratio(
                        ratios,
                        split,
                        metric=metric,
                        intervention_distrib=config.intervention_distrib,
                        intervention_bank=intervention_bank,
                        edge_scores=edge_scores[split],
                        graphs=graphs[split],
                        graphs_nx=graphs_nx[split],
                        labels=labels[split],
                        avg_graph_size=avg_graph_size[split],
                        causal_subgraphs_r=causal_subgraphs_r[split],
                        spu_subgraphs_r=spu_subgraphs_r[split],
                        expl_accs_r=expl_accs_r[split],
                        causal_masks_r=causal_masks_r[split]
                    )
                    # assign_dict(
                    #     results_big,
                    #     [expname, split, metric, f"seed_{seed}"],
                    #     score
                    # )
                    metrics_score[load_split][split][metric].append(score)
                    metrics_score[load_split][split][metric + "_acc_int"].append(acc_int)
        
        # if not "suff" in args.metrics and not "acc" in args.metrics:
        #     print("\n\n")
        #     for split in splits:
        #         avg_score = {}
        #         for metric_key in results_big[expname][split][metric][f"seed_{args.seeds[0]}"].keys():
        #             sa = []
        #             for seed in results_big[expname][split][metric].keys():
        #                 sa.append(results_big[expname][split][metric][seed][metric_key])
        #             avg_score[metric_key] = np.mean(sa, axis=0)
        #         print(f"Manually averaged results ({split}): ", avg_score)
        #         print("\n\n")
        #         assign_dict(
        #             results_big,
        #             [expname, split, metric, "seed_avg"],
        #             avg_score
        #         )
        #         print(results_big[expname][split]["nec"])
        #         print("\n\n")

        # if metric.lower() in ("suff", "suff++" "nec", "nec++", "fidp", "fidm") and config.save_metrics:
        #     with open(f"storage/metric_results/{load_split}_results.json", "w") as f:
        #         json.dump(results_big, f)        

    if config.save_metrics:
        save_path = f"storage/metric_results/aggregated_{load_split}_results_necalpha{config.nec_alpha_1}" \
                    f"_numsamples{config.numsamples_budget}_randomexpl{config.random_expl}_ratios{args.ratios.replace('/','-')}" \
                    f"_metrics{args.metrics.replace('/','-')}" \
                    f"_{config.log_id}.json"
        if not os.path.exists(save_path):
            with open(save_path, 'w') as file:
                file.write("{}")
        with open(save_path, "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    else:
        results_aggregated = None

    for load_split in load_splits:
        print("\n\n", "-"*50, f"\nPrinting evaluation results for load_split {load_split}\n\n")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                print(f"{metric} = {metrics_score[load_split][split][metric]}")
        if "acc" in args.metrics.split("/"):
            for split in splits + ["test", "test_R"]:
                print(f"\nEval split {split}")
                for metric in ["acc"]: #, "plaus", "wiou"
                    print(f"{metric} = {metrics_score[load_split][split][metric]}")

        if "plaus" in args.metrics:
            print("\n\n", "-"*50, "\nComputing Plausibility")
            for split in splits:
                print(f"\nEval split {split}")
                for div in ["wiou", "F1"]:
                    s = metrics_score[load_split][split][div]
                    print_metric(div, s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, div])
            continue

        print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                if "acc" == metric:
                    continue
                # for c in metrics_score[load_split][split][metric][0].keys():
                #     s = [
                #         metrics_score[load_split][split][metric][i][c] for i in range(len(metrics_score[load_split][split][metric]))
                #     ]
                #     print_metric(metric + f" class {c}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, metric])
                for div in ["L1", "KL"]:
                    s = [
                        metrics_score[load_split][split][metric][i][f"all_{div}"] for i in range(len(metrics_score[load_split][split][metric]))
                    ]
                    print_metric(metric + f" class all_{div}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, metric+f"_{div}"])
                print(metrics_score[load_split][split][metric + "_acc_int"])
                print(s)
                print_metric(metric + "_acc_int", metrics_score[load_split][split][metric + "_acc_int"], results_aggregated, key=[config.dataset.dataset_name+" "+config.dataset.domain, config.complete_dirname, split, metric+"_acc_int"])
                
        if "acc" in args.metrics.split("/"):
            for split in splits + ["test", "test_R"]:
                print(f"\nEval split {split}")
                print_metric("acc", metrics_score[load_split][split]["acc"])
                for a in ["plaus", "wiou"]:
                    for c in metrics_score[load_split][split][a][0].keys():
                        s = [
                            metrics_score[load_split][split][a][i][c] for i in range(len(metrics_score[load_split][split][a]))
                        ]
                        print_metric(a + f" class {c}", s)

        print("\n\n", "-"*50, "\nComputing faithfulness")
        for split in splits:
            print(f"\nEval split {split}")            
            for div in ["L1", "KL"]:
                if "suff" in args.metrics.split("/") and "nec" in args.metrics.split("/"):                
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_gmean_{div}"])

                if "suff++" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff++"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_gmean_{div}"])
                
                if "suff_simple" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff_simple"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_gmean_{div}"])

                if "suff" in args.metrics.split("/") and "nec++" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
                    necpp = get_tensorized_metric(metrics_score[load_split][split]["nec++"], f"all_{div}")[:, :suff.shape[1]]
                    faith_aritm = aritm(suff, necpp)
                    faith_armonic = armonic(suff, necpp)
                    faith_gmean = gmean(suff, necpp)
                    print_metric(f"Faith.++ Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith++_aritm_{div}"])
                    print_metric(f"Faith.++ Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith++_gmean_{div}"])
                    print_metric(f"Faith.++ GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith++_gmean_{div}"])

        print(f"Computed for split load_split = {load_split}\n\n\n")
    
    if config.save_metrics:
        with open(save_path, "w") as f:
            json.dump(results_aggregated, f)     
    
    print("Completed in ", datetime.now() - startTime, f" for {config.complete_dirname} {config.dataset.dataset_name}/{config.dataset.domain}")
    print("\n\n")
    sys.stdout.flush()

def print_faith(args):
    load_split = "id"
    config = config_summoner(args)

    split_metrics = ["id_val", "val", "test"]
    metrics = ["suff_simple_L1", "nec_L1", "faith_armon_L1"]
    model = config.complete_dirname
    dataset = config.dataset.dataset_name + " " + config.dataset.domain

    print("\nMODEL = \t", model)
    print("DATASET = \t", dataset)
    print("\n\n")

    results = {
        True: {split: {} for split in split_metrics},
        False: {split: {} for split in split_metrics},
    }
    big_rows = {s: "" for s in split_metrics}
    for split_metric in split_metrics:
        print(f"{split_metric}")
        for j, metric in enumerate(metrics):
            for i, random_expl in enumerate([False, True]):
                save_path = f"storage/metric_results/aggregated_{load_split}_results_necalpha{config.nec_alpha_1}" \
                            f"_numsamples{config.numsamples_budget}_randomexpl{random_expl}_ratios{args.ratios.replace('/','-')}" \
                            f"_metrics{args.metrics.replace('/','-')}" \
                            f"_{config.log_id}.json"
                if split_metric == split_metrics[0] and i == 0 and j == 0:
                    print("METRIC FILE = \t", save_path)

                with open(save_path, "r") as jsonFile:
                    data = json.load(jsonFile)
                if not model in data[dataset].keys() or not split_metric in data[dataset][model].keys():
                    continue
                
                results[random_expl][split_metric][f"values_{metric}"]     = [f"{d:.2f}" for d in data[dataset][model][split_metric][metric]]
                results[random_expl][split_metric][f"stds_{metric}"]       = [f"{d:.2f}" for d in data[dataset][model][split_metric][metric + "_std"]]

                if config.log_id == "isweight" and len(results[random_expl][split_metric][f"values_{metric}"]) == 4:
                    # append an empty entry for easy formatting on Sheet
                    results[random_expl][split_metric][f"values_{metric}"].append("-")
                    results[random_expl][split_metric][f"stds_{metric}"].append("-")

                row = "; ".join([f"{v} +- {s}" for v,s in zip(results[random_expl][split_metric][f"values_{metric}"], results[random_expl][split_metric][f"stds_{metric}"])])
                
                print(f"\t{metric + (' Rnd' if random_expl else ''):<25}:\t{row}")
                big_rows[split_metric] = big_rows[split_metric] + ";" + row

                if "faith" in metric and random_expl == True:
                    ratio_values = [ f"{float(results[True][split_metric][f'values_{metric}'][k]) / float(results[False][split_metric][f'values_{metric}'][k]):.2f}" for k in range(len(data[dataset][model][split_metric][metric]))]
                    
                    # ratio_stds = [ f"0.00" for _ in range(len(data[dataset][model][split_metric][metric]))]
                    # Z = Y/X = Rnd/Orig
                    # Computed using the Delta method (assuming R.V. asymptotically Gaussian and X and Y independent)
                    # https://stats.stackexchange.com/questions/291594/estimation-of-population-ratio-using-delta-method
                    # http://www.senns.uk/Stats_Notes/Variance_of_a_ratio.pdf
                    ratio_vars = [
                            float(results[True][split_metric][f'stds_{metric}'][k])**2 / float(results[False][split_metric][f'values_{metric}'][k])**2 + \
                                (float(results[True][split_metric][f'values_{metric}'][k])**2 * float(results[False][split_metric][f'stds_{metric}'][k])**2 / float(results[False][split_metric][f'values_{metric}'][k])**4)
                        for k in range(len(data[dataset][model][split_metric][metric]))
                    ]
                    ratio_stds = [ f"{d**0.5:.2f}" for d in ratio_vars]


                    row = "; ".join([f"{v} +- {s}" for v,s in zip(ratio_values, ratio_stds)])
                    
                    print(f"\t{metric + ' ratio':<25}:\t{row}")
                    big_rows[split_metric] = big_rows[split_metric] + ";" + row

    print("\n\nPrinting big rows:\n")
    for split_metric in split_metrics:
        print(f"{split_metric}: {big_rows[split_metric]}\n\n")

def evaluate_metric(args):
    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val"] #"id_val", "val", "test"
    print("Using splits = ", splits)
        
    if args.ratios != "":
        thrs = [float(r) for r in args.ratios.split("/")]
    else:
        thrs = [0.1]
    print("Using thresholds = ", thrs)
    
    startTime = datetime.now()
    metrics_score = {s: defaultdict(list) for s in splits}
    for i, seed in enumerate(args.seeds.split("/")):
        ##
        # SET UP THE CONFIGURATION AND LOAD MODEL
        ##
        seed = int(seed)        
        args.random_seed = seed
        args.exp_round = seed
        
        config = config_summoner(args)
        config["task"] = "test"
        config["load_split"] = ""
        if i == 0:
            load_logger(config)
        
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
        print("DEBUG BEFORE load_task")
        pipeline.load_task(load_param=True, load_split="id")
        print("DEBUG AFTER load_task")
        
        ##
        # GENERATE BINARY EXPLANATION MASKS
        ##
        samples, graphs_nx, avg_graph_size  = pipeline.generate_binary_explanations(
            is_weight=True, 
            thrs=thrs,
            splits=splits,
            convert_to_nx="interven_suff" in args.metrics,
            is_node_expl=(
                True
                if config.model.model_name == "GSATGNNs_MODIFIED"
                else not config.ood.extra_param[0]
            ) # GSATGNNs_MODIFIED returns node explanations
        )            

        for metric in args.metrics.split("/"):
            print(f"\n\nEvaluating {metric.upper()} for seed {seed}\n")

            for split in splits:
                for thr in thrs:
                    print(f"\n\n#D#Computing {metric.upper()} over {split} across ratios with thr={thr}")
                    score, acc_int = pipeline.compute_metric(
                        metric=metric,
                        graphs=samples[split][thr],
                        graphs_nx=graphs_nx[split],
                        avg_graph_size=avg_graph_size[split],
                    )
                    metrics_score[split][metric].append(score)
                    metrics_score[split][metric + "_acc_int"].append(acc_int)


    if config.save_metrics:
        save_path = f"storage/metric_results/aggregated_id_results_necalpha{config.nec_alpha_1}" \
                    f"_numsamples{config.numsamples_budget}_randomexpl{config.random_expl}_ratios{args.ratios.replace('/','-')}" \
                    f"_metrics{args.metrics.replace('/','-')}" \
                    f"_{config.log_id}.json"
        if not os.path.exists(save_path):
            with open(save_path, 'w') as file:
                file.write("{}")
        with open(save_path, "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    else:
        results_aggregated = None

    print("\n\n", "-"*50, f"\nPrinting evaluation results\n\n")
    for split in splits:
        print(f"\nEval split {split.upper()}")
        for metric in args.metrics.split("/"):
            print(f"{metric} = {metrics_score[split][metric]}")

    print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
    for split in splits:
        print(f"\nEval split {split.upper()}")
        for metric in args.metrics.split("/"):
            for div in ["predicted"]:
                for c in range(10):
                    if f"{c}_{div}" not in metrics_score[split][metric][i].keys():
                        continue
                    # take values acorss seed, then print them
                    s = [
                        metrics_score[split][metric][i][f"{c}_{div}"] for i in range(len(metrics_score[split][metric]))
                    ]
                    print_metric(metric + f" class {c}_{div}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, metric+f"_{div}"])    
                s = [
                    metrics_score[split][metric][i][f"all_{div}"] for i in range(len(metrics_score[split][metric]))
                ]
                print_metric(metric + f" class all_{div}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, metric+f"_{div}"])
            print_metric(metric + "_acc_int", metrics_score[split][metric + "_acc_int"], results_aggregated, key=[config.dataset.dataset_name+" "+config.dataset.domain, config.complete_dirname, split, metric+"_acc_int"])
            
            s = [
                metrics_score[split][metric][i][f"rejection"] for i in range(len(metrics_score[split][metric]))
            ]
            print_metric(metric + f" rejection", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, metric+f"_rejection"])
            print()

    # print("\n\n", "-"*50, "\nComputing faithfulness")
    # for split in splits:
    #     print(f"\nEval split {split}")            
    #     for div in ["L1", "KL"]:
    #         if "suff" in args.metrics.split("/") and "nec" in args.metrics.split("/"):                
    #             suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
    #             nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
    #             faith_aritm = aritm(suff, nec)
    #             faith_armonic = armonic(suff, nec)
    #             faith_gmean = gmean(suff, nec)
    #             print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_aritm_{div}"])
    #             print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_armon_{div}"])
    #             print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_gmean_{div}"])

    #         if "suff++" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
    #             suff = get_tensorized_metric(metrics_score[load_split][split]["suff++"], f"all_{div}")
    #             nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
    #             faith_aritm = aritm(suff, nec)
    #             faith_armonic = armonic(suff, nec)
    #             faith_gmean = gmean(suff, nec)
    #             print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_aritm_{div}"])
    #             print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_armon_{div}"])
    #             print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_gmean_{div}"])
            
    #         if "suff_simple" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
    #             suff = get_tensorized_metric(metrics_score[load_split][split]["suff_simple"], f"all_{div}")
    #             nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
    #             faith_aritm = aritm(suff, nec)
    #             faith_armonic = armonic(suff, nec)
    #             faith_gmean = gmean(suff, nec)
    #             print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_aritm_{div}"])
    #             print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_armon_{div}"])
    #             print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_gmean_{div}"])

    #         if "suff" in args.metrics.split("/") and "nec++" in args.metrics.split("/"):
    #             suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
    #             necpp = get_tensorized_metric(metrics_score[load_split][split]["nec++"], f"all_{div}")[:, :suff.shape[1]]
    #             faith_aritm = aritm(suff, necpp)
    #             faith_armonic = armonic(suff, necpp)
    #             faith_gmean = gmean(suff, necpp)
    #             print_metric(f"Faith.++ Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith++_aritm_{div}"])
    #             print_metric(f"Faith.++ Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith++_gmean_{div}"])
    #             print_metric(f"Faith.++ GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith++_gmean_{div}"])
    
    # if config.save_metrics:
    #     with open(save_path, "w") as f:
    #         json.dump(results_aggregated, f)     
    
    print("\nCompleted in ", datetime.now() - startTime, f" for {config.complete_dirname} {config.dataset.dataset_name}/{config.dataset.domain}")
    print("\n\n")

def print_r_ge_b_hist(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        edge_scores_seed = []
        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING PLOT FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            if config.dataset.dataset_name in ("BAColor", "BAColorGV", "BAColorGVIsolated"):
                print(f"\n\nClassifier weights:")
                print(model.classifierS.classifier[0].weight.detach()) #, model.classifier.classifier[0].bias.detach()

            # GET EXPLANATIONS
            ret = pipeline.get_node_explanations()

            # AGGREGATE INFO BY LABEL
            list_of_labels = np.array([ret["id_val"]["samples"][i].y.item() for i in range(len(ret["id_val"]["samples"]))])
            list_of_colors = {l: [] for l in np.unique(list_of_labels)}
            count_of_relevant_colors = {l: defaultdict(list) for l in np.unique(list_of_labels)}
            list_of_scores = {l: [] for l in np.unique(list_of_labels)}

            if "DIR" in config.model.model_name:
                importance_threshold = 0.05
            else:
                importance_threshold = 0.5

            for i, label in enumerate(np.unique(list_of_labels)):
                for j in range(len(ret["id_val"]["samples"])):
                    if ret["id_val"]["samples"][j].y.item() == label:
                        node_colors = list(
                            map( # Convert list of tuples into list of colors
                                loader["id_val"].dataset.color_map2.get,
                                [int(r) for r in ret["id_val"]["samples"][j].x.argmax(dim=1)] # convert feature matrix into list of tuples [(1., 0.)]
                            )
                        )
                        list_of_colors[label].extend(node_colors)
                        list_of_scores[label].extend(
                            ret["id_val"]["scores"][j]
                        )

                        important_colors_count = np.unique(
                            np.array(node_colors)[np.array(ret["id_val"]["scores"][j]) >= importance_threshold],
                            return_counts=True
                        )

                        for color in np.unique(node_colors):
                            if color in important_colors_count[0]:
                                count_of_relevant_colors[label][color].append(important_colors_count[1][important_colors_count[0] == color][0])
                            else:
                                count_of_relevant_colors[label][color].append(0)

                        # if label == 1.0 and count_of_relevant_colors[label]["R"] >= count_of_relevant_colors[label]["B"]:
                        #     print(i)

                # average the count of relevant colors
                # for c in count_of_relevant_colors[label].keys():
                #     count_of_relevant_colors[label][c] /= sum(loader["id_val"].dataset.y == label).item()

                list_of_colors[label] = np.array(list_of_colors[label])
                list_of_scores[label] = np.array(list_of_scores[label])

            
            # PLOT HISTOGRAMS
            n_row = 2
            n_col = np.unique(list_of_colors[0]).shape[0] + 1
            fig, axs = plt.subplots(n_row, n_col, figsize=(12,7))

            for i, label in enumerate(np.unique(list_of_labels)):
                print(f"\ny={int(label)}")
                print(f"\tStats of node scores: min={min(list_of_scores[label]):.3f}, max={max(list_of_scores[label]):.3f}, avg={np.mean(list_of_scores[label]):.3f}, std={np.std(list_of_scores[label]):.3f} ")
                print(f"\tAverage count of relevant colors:", {c: round(np.mean(count_of_relevant_colors[label][c]), 2) for c in count_of_relevant_colors[label].keys()})
                
                # per-color hist
                for c, color in enumerate(["R", "B", "G", "V"]):
                    axs[i,c].hist(
                        list_of_scores[label][list_of_colors[label] == color] + np.random.normal(0.0, scale=0.005, size=list_of_scores[label][list_of_colors[label] == color].shape),
                        density=True,
                        log=False,
                        bins=100,
                        label=color
                    )
                    if "DIR" in config.model.model_name:
                        axs[i,c].set_xlim(-1.1, 1.1)
                    else:
                        axs[i,c].set_xlim(-0.1, 1.1)
                    axs[i,c].set_ylim(0.0, 100)
                    axs[i,c].set_title(f"color {color}")
                    if c == 0:
                        axs[i,0].set_ylabel(f"y={int(label)}")
                
                # per-sample boxplot
                axs[i, -1].set_title(f"per-sample avg count")
                bplot = axs[i, -1].boxplot([
                        count_of_relevant_colors[label][c] for c in ["R", "B", "G", "V"]
                    ],
                    patch_artist=True,
                    labels=["R", "B", "G", "V"],
                    showfliers=False,
                    showmeans=True
                )
                for patch, color in zip(bplot['boxes'], ["red", "blue", "green", "violet"]):
                    patch.set_facecolor(color)
                axs[i, -1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

            fig.supxlabel('explanation relevance scores', fontsize=13)
            fig.supylabel('density', fontsize=13)
            fig.suptitle(f'{config.model.model_name} seed {seed}', fontsize=13)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            path = f'{ROOT_DIR}/GOOD/kernel/pipelines/plots/panels/{config.ood_dirname}/'
            if not os.path.exists(path):
                os.makedirs(path)
            path += f"{config.load_split}_{config.dataset.dataset_name}_{config.dataset.domain}_{config.util_model_dirname}_{config.random_seed}"
            plt.savefig(path + ".png")
            plt.close()
            print("File saved in ", path)

def print_hist(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        edge_scores_seed = []
        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING PLOT FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            if config.dataset.dataset_name in ("BAColor", "BAColorGV", "BAColorGVIsolated"):
                print(f"\n\nClassifier weights:")
                print(model.classifierS.classifier[0].weight.detach()) #, model.classifier.classifier[0].bias.detach()

            # GET EXPLANATIONS
            ret = pipeline.get_node_explanations()

            # AGGREGATE INFO BY LABEL
            list_of_labels = np.array([ret["id_val"]["samples"][i].y.item() for i in range(len(ret["id_val"]["samples"]))])
            list_of_scores = []

            for j in range(len(ret["id_val"]["samples"])):
                expl = ret["id_val"]["scores"][j]
                
                # if "SMGNN" in config.model.model_name and "MNIST" in config.dataset.dataset_name:
                #     expl = (np.array(expl) - min(expl)) / (max(expl) - min(expl))
                #     expl = expl.tolist()

                list_of_scores.extend(
                    expl
                )
            list_of_scores = np.array(list_of_scores)
            
            # PLOT HISTOGRAMS
            n_row = 1
            n_col = 1
            fig, axs = plt.subplots(n_row, n_col, figsize=(12,3.5*n_row))

            axs.hist(
                list_of_scores, #np.random.normal(0.0, scale=0.005, size=list_of_scores.shape)
                density=True,
                log=False,
                bins=100
            )
            if "DIR" in config.model.model_name:
                axs.set_xlim(-1.1, 1.1)
            else:
                axs.set_xlim(-0.1, 1.1)
            # axs.set_ylim(0.0, 100)                

            fig.supxlabel('explanation relevance scores', fontsize=13)
            fig.supylabel('density', fontsize=13)
            fig.suptitle(f'{config.model.model_name} seed {seed}', fontsize=13)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            path = f'{ROOT_DIR}/GOOD/kernel/pipelines/plots/panels/{config.ood_dirname}/'
            if not os.path.exists(path):
                os.makedirs(path)
            path += f"{config.load_split}_{config.dataset.dataset_name}_{config.dataset.domain}_{config.util_model_dirname}_{config.random_seed}"
            print("Saving at ", path)
            plt.savefig(path + ".png")
            plt.close()

def plot_explanations(args):
    load_splits = ["id"]
    split = "id_val"
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING PLOT FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            if config.dataset.dataset_name in ("BAColor", "BAColorGV", "BAColorGVIsolated"):
                print(f"\n\nClassifier weights:")
                print(model.classifierS.classifier[0].weight.detach())

            normalize = False
            if "SMGNN" in config.model.model_name and config.dataset.dataset_name == "MNIST":
                print("\n\nNormalizing scores in [0,1]\n\n")
                normalize = True

            # GET EXPLANATIONS
            N = 20
            ret = pipeline.get_node_explanations(num_samples=N)

            # DEFINE IMPORTANCE THRESHOLD
            if "DIR" in config.model.model_name:
                thr = 0.0 # just filter based on topK
            else:
                thr = 0.5

            # PLOT GRAPHS
            compute_fid = False
            topK_nodes_kept = None
            for i in range(len(ret[split]["samples"])):
                data = ret[split]["samples"][i].cpu()
                expl = ret[split]["scores"][i]                

                if normalize:
                    expl = (np.array(expl) - min(expl)) / (max(expl) - min(expl))

                data.node_expl = torch.tensor(expl, device=data.x.device)
                data.node_mask = data.node_expl >= thr
                data.edge_mask = torch.ones_like(data.edge_index[0])

                # data = Batch().from_data_list([data])
                # graph_label_per_node = data.y[data.batch]
                # num_max_sp_per_batch = scatter_max(data.sp_order, index=data.batch)[0][data.batch]
                # targets = torch.logical_or(data.sp_order == graph_label_per_node, data.sp_order == (num_max_sp_per_batch - graph_label_per_node))
                # data.x[targets, :3] = torch.tensor([1,0,0], device=data.x.device, dtype=data.x.dtype)
                
                if "MNIST" in config.dataset.dataset_name:
                    pred = ret[split]["pred"][i].argmax(dim=0)
                else:
                    if config.metric.dataset_task == 'Multi-label classification':
                        pred = round(ret[split]["pred"][i].softmax(dim=0)[1].item(), 3)
                    else:
                        pred = round(ret[split]["pred"][i].sigmoid().item(), 3)

                # title = f"Idx: {i:<3} Class={int(data.y.item())} Pred={pred:<5}"
                title = f"Class={int(data.y.item())} Prediction={pred:<5}"

                if "DIR" in config.model.model_name:
                    if i == 0:
                        print(f"Highlightinh nodes in the Top-{config.ood.ood_param}%")

                    # Highlight TopK nodes
                    (causal_x, causal_edge_index, causal_edge_attr, causal_batch, causal_node_weight), \
                        (conf_x, conf_edge_index, conf_edge_attr, conf_batch, conf_node_weight), \
                            (topK_nodes_kept, topK_nodes_removed) = split_graph_node(data, torch.tensor(expl, device=data.x.device), config.ood.ood_param, embed=None, use_input_feat=True)
                    topK_nodes_kept = topK_nodes_kept.cpu().tolist()

                    # remove nodes not in the TopK
                    assert len(topK_nodes_kept) + topK_nodes_removed.shape[0] == data.x.shape[0]
                    data.node_mask[topK_nodes_removed] = False

                if compute_fid:
                    for metric in ["fidm", "fidp", "rfidm", "rfidp", "suff", "suff_cause"]:
                        val = pipeline.compute_metric(metric=metric, graphs=[data], graphs_nx=None, avg_graph_size=None, log_info=False)[0]["all_predicted"][0]
                        title += f" {metric}={val:.2f}"
                    # fidm = pipeline.compute_metric(metric="fidm", graphs=[data], graphs_nx=None, avg_graph_size=None, log_info=False)[0]["all_predicted"][0]
                    # fidp = pipeline.compute_metric(metric="fidp", graphs=[data], graphs_nx=None, avg_graph_size=None, log_info=False)[0]["all_predicted"][0]
                    # suff_cause = pipeline.compute_metric(metric="suff_cause", graphs=[data], graphs_nx=None, avg_graph_size=None, log_info=False)[0]["all_predicted"][0]
                    # title += f" FIDM={fidm:.2f} FIDP={fidp:.2f} SUFF_CAUSE={suff_cause:.2f}"

                if "SST2" in config.dataset.dataset_name:
                    xai_utils.plot_sentence_graph(
                        G=data,
                        name=f"graph_{split}_{i}",
                        subfolder=f"plots_of_explanation_examples/{config.prefix}{config.ood_dirname}/{config.dataset.dataset_name}_{config.dataset.domain}",
                        config=config,
                        title=title,
                    )
                else:
                    g = to_networkx(data, node_attrs=["x", "node_mask"], to_undirected=True)
                    xai_utils.draw_colored(
                        config,
                        g,
                        node_expl=expl,
                        subfolder=f"plots_of_explanation_examples/{config.prefix}{config.ood_dirname}/{config.dataset.dataset_name}_{config.dataset.domain}",
                        name=f"graph_{split}_{i}",
                        title=title,
                        with_labels=False,
                        figsize=(12,10) if "AIDS" in config.dataset.dataset_name else (6.4, 4.8),
                        topk=topK_nodes_kept
                    )
                print(f"graph {title}")
            print("Plotted in ", f"plots_of_explanation_examples/{config.prefix}{config.ood_dirname}/{config.dataset.dataset_name}_{config.dataset.domain}")
