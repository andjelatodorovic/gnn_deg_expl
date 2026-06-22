r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import json
from collections import defaultdict
from datetime import datetime

from torch_geometric import __version__ as __pyg_version__

from GOOD import config_summoner
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.logger import load_logger
from GOOD.utils.metric import assign_dict
from GOOD.definitions import OOM_CODE
import GOOD.utils.analysis as analysis
from GOOD.utils.loader import initialize_model_dataset


import warnings
import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.stats import pearsonr

from torch import set_num_threads
set_num_threads(6)

# if __pyg_version__ == "2.4.0":
#     torch.set_num_threads(6)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    args = args_parser()

    assert not args.seeds is None, args.seeds

    if args.task == 'eval_metric':
        analysis.evaluate_metric(args)
        exit(0)
    if args.task == 'plot_panel':
        analysis.generate_panel(args)
        exit(0)
    if args.task == 'permute_attention':
        analysis.permute_attention_scores(args)
        exit(0)
    if args.task == 'plot_sampling':
        analysis.generate_plot_sampling(args)
        exit(0)
    if args.task == 'stability_detector':
        # stability_detector_rebuttal(args)
        analysis.stability_detector_extended(args)
        exit(0)
    if args.task == 'plot_explanations':
        # analysis.plot_explanation_examples(args)
        analysis.plot_explanations(args)
        exit(0)
    if args.task == 'print_faith':
        analysis.print_faith(args)
        exit(0)
    if args.task == 'hist':
        # analysis.print_r_ge_b_hist(args)
        analysis.print_hist(args)
        exit(0)

    run = None
    test_scores, test_losses, ckpt_losses = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod, test_auroc = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    test_f1_pos, test_f1_neg = defaultdict(list), defaultdict(list)
    test_prec_pos, test_prec_neg, test_recall_pos, test_recall_neg = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for i, seed in enumerate(args.seeds.split("/")):
        seed = int(seed)
        print(f"\n\n#D#Running with seed = {seed}")
        
        args.random_seed = seed
        args.exp_round = seed
        
        config = config_summoner(args)
        print(config.random_seed, config.exp_round)
        print(args)
        if i == 0:
            load_logger(config)
        
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)

        if config.task == 'train':
            startTrainTime = datetime.now()
            if config.wandb:
                run = wandb.init(
                    project="explanation-shortcuts",
                    config=config,
                    entity="mcstewe",
                    name=f'{config.dataset.dataset_name}_{config.dataset.domain}{config.ood_dirname}_{config.util_model_dirname}_{config.random_seed}'
                )
                wandb.watch(pipeline.model, log="all", log_freq=10)

            # Train model
            pipeline.load_task()            
            print(f'\nTraining end ({datetime.now() - startTrainTime}).\n')

            # Eval model
            pipeline.task = 'test'
            test_score, ckpt = pipeline.load_task(load_param=True, load_split="id")
            test_scores["saved_score"].append(test_score)
            for s in ["train", "id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(s, epoch=ckpt["epoch"])
                test_scores[s].append(sa['score'])
                test_auroc[s].append(sa['aucroc'])
                for l in ("spec_loss", "entr_loss", "l_norm_loss", "clf_loss", "total_loss"):
                    test_losses[s][l].append(sa["loss_dict"][l])
        elif config.task == 'test':
            compute_clf_only_pred = False #(s == "val"),
            test_score, ckpt = pipeline.load_task(load_param=True, load_split="id")

            for s in ["train", "id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(
                    s,
                    compute_plaus=False,
                    compute_mcc=config.train.pretrain == "degenerate" or config.train.pretrain == "suff",
                    compute_clf_only_pred=compute_clf_only_pred, 
                    epoch=ckpt["epoch"]
                )
                test_scores[s].append(sa['score'])
                # test_losses[s].append(sa['loss'].item())
                test_auroc[s].append(sa['aucroc'])
                test_f1_pos[s].append(sa['f1_pos'])
                test_f1_neg[s].append(sa['f1_neg'])
                test_prec_pos[s].append(sa['prec_pos'])
                test_recall_pos[s].append(sa['recall_pos'])
                test_prec_neg[s].append(sa['prec_neg'])
                test_recall_neg[s].append(sa['recall_neg'])
                test_likelihoods_avg[s].append(sa['likelihood_avg'].item())
                test_likelihoods_prod[s].append(sa['likelihood_prod'].item())
                test_likelihoods_logprod[s].append(sa['likelihood_logprod'].item())
                for l in ("spec_loss", "entr_loss", "l_norm_loss", "clf_loss", "total_loss"):
                    test_losses[s][l].append(sa["loss_dict"][l])

                if compute_clf_only_pred:
                    print("Predictions on VAL: \t", sa['pred'][0])
                    print("CLF Predictions VAL: \t", sa['pred_clf_only'][0])

            for loss_name in ckpt.keys():
                if loss_name in ("spec_loss", "total_loss", "entr_loss", "l_norm_loss", "clf_loss"):
                    ckpt_losses[loss_name].append(ckpt[loss_name])
    
    if config.save_metrics:
        with open(f"storage/metric_results/acc_plaus.json", "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    
    print("\n\nFinal accuracies: ")
    for s in test_scores.keys():
        print(f"{s.upper():<10} = {np.mean(test_scores[s]):.3f} +- {np.std(test_scores[s]):.3f}")

    if not np.isnan(test_auroc["train"][0]):
        print("\n\nFinal AUCROC: ")
        for s in test_auroc.keys():
            print(f"{s.upper():<10} = {np.mean(test_auroc[s]):.3f} +- {np.std(test_auroc[s]):.3f}")

    if (
        "train" in test_f1_pos
        and len(test_f1_pos["train"]) > 0
        and not np.isnan(test_f1_pos["train"][0])
    ):
        print("\n\nFinal F1_pos: ")
        for s in test_f1_pos.keys():
            print(f"{s.upper():<10} = {np.mean(test_f1_pos[s]):.3f} +- {np.std(test_f1_pos[s]):.3f}")
        print("\n\nFinal F1_neg: ")
        for s in test_f1_neg.keys():
            print(f"{s.upper():<10} = {np.mean(test_f1_neg[s]):.3f} +- {np.std(test_f1_neg[s]):.3f}")
        print("\n\nMacro F1: ")        
        for s in test_f1_neg.keys():
            macros = np.mean(np.concatenate(([test_f1_pos[s]], [test_f1_neg[s]]), axis=0), axis=0)
            print(f"{s.upper():<10} = {np.mean(macros):.3f} +- {np.std(macros):.3f}")
        # print("\n\nFinal Prec_pos: ")
        # for s in test_prec_pos.keys():
        #     print(f"{s.upper():<10} = {np.mean(test_prec_pos[s]):.3f} +- {np.std(test_prec_pos[s]):.3f}")
        # print("\n\nFinal Recall_pos: ")
        # for s in test_prec_pos.keys():
        #     print(f"{s.upper():<10} = {np.mean(test_recall_pos[s]):.3f} +- {np.std(test_recall_pos[s]):.3f}")
        # print("\n\nFinal Prec_neg: ")
        # for s in test_prec_pos.keys():
        #     print(f"{s.upper():<10} = {np.mean(test_prec_neg[s]):.3f} +- {np.std(test_prec_neg[s]):.3f}")
        # print("\n\nFinal Recall_neg: ")
        # for s in test_prec_pos.keys():
        #     print(f"{s.upper():<10} = {np.mean(test_recall_neg[s]):.3f} +- {np.std(test_recall_neg[s]):.3f}")


    print("\n\nFinal losses: ")
    for s in test_losses.keys():
        string = f"{s.upper():<10}"
        for l in ("total_loss", "spec_loss", "entr_loss", "l_norm_loss", "clf_loss"):
            if not np.isnan(np.mean(test_losses[s][l])):
                string += f"  {l:<8} {np.mean(test_losses[s][l]):.4f} +- {np.std(test_losses[s][l]):.4f}"
        print(string)

    print("\n\nCkpt losses for TRAIN: ")
    for s in ckpt_losses.keys():
        print(f"{s:<12} = {np.mean(ckpt_losses[s]):.4f} +- {np.std(ckpt_losses[s]):.4f}")

    if config.save_metrics:
        print("Saving metrics to json...")
        for s in test_losses.keys():
            for name, d in zip(
                ["loss_entiresplit", "likelihood_avg_entiresplit", "likelihood_prod_entiresplit", "likelihood_logprod_entiresplit"], 
                [test_losses, test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod]
            ):
                key = [config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, s, name]
                if s in results_aggregated[key[0]][key[1]].keys():            
                    assign_dict(
                        results_aggregated,
                        key,
                        np.mean(d[s])
                    )
                    key[-1] += "_std"
                    assign_dict(
                        results_aggregated,
                        key,
                        np.std(d[s])
                    )
        with open(f"storage/metric_results/acc_plaus.json", "w") as f:
            json.dump(results_aggregated, f)  

    if config.wandb and run:
        run.finish()

def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e


if __name__ == '__main__':
    main()
