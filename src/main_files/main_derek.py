import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# the stuff above is for path finding to the proper imports, don't worry about it too much


from dataset_manager import DatasetManager
from model import GarbageClassificationModel
from transfer_learning_models.resnet50_model import ResNet50Model
from transfer_learning_models.mobilenetv3_model import MobileNetV3Model
from transfer_learning_models.nasnetmobile_model import NASNetMobileModel

from logger import Logger



if __name__ == "__main__":
    

    DATASET_DIR = "datasets/dataset_vers2/"

    dm = DatasetManager("datasets/dataset_vers1/", (224, 224))
    print(dm.get_categories())

    dm = DatasetManager("datasets/dataset_vers2/", (224, 224))
    print(dm.get_categories())

    # model_classes = [MobileNetV3Model]
    # image_sizes = [224] # decided on 224x224 for the image size (it works the best)
    # min_dim = 50
    
    # for model_class in model_classes:
    #     for image_dim in image_sizes:
    #         img_size = (image_dim, image_dim)
    #         export_name = f"FINAL_VER_{model_class.__name__}_datasetvers2_imgsize_{img_size}_mindim_({min_dim})"
    #         logger = Logger(export_name)
    #         dl = DatasetManager(DATASET_DIR, img_size, logger=logger)
    #         m = model_class(dl, logger=logger)
    #         dl.load_data(min_dim=min_dim)
    #         dl.split_data(60, 20, 20)
    #         model_history = m.train_model(
    #             epochs=25,
    #             batch_size=32,
    #             export_path=f"model_registry/{export_name}.keras"
    #         )
    #         m.plot_history(model_history)
    #         m.measure_metrics()
    #         m.predict_img("datasets/dataset_vers2/organic/organic_1.jpg")
    #         m.random_preds(3, 6)
    #         m.random_preds(8, 4)


    # TEST 1:  testing which image size to normalize to for the model to ingest
    # ================================================================ TRAINING THE MODELS ================================================================
    
    # DATASET_DIR = "datasets/dataset_vers2/"

    # model_classes = [MobileNetV3Model]
    # image_sizes = [224]
    
    # for model_class in model_classes:
    #     for image_dim in image_sizes: # decided on 224x224 for the image size (it works the best)
    #         img_size = (image_dim, image_dim)
    #         export_name = f"{model_class.__name__}_imgsize_{img_size}"
    #         logger = Logger(export_name)
    #         dl = DatasetManager(DATASET_DIR, img_size, logger=logger)
    #         m = model_class(dl, logger=logger)
    #         dl.load_data(min_dim=100)
    #         dl.split_data(60, 20, 20)
    #         model_history = m.train_model(
    #             epochs=25,
    #             batch_size=32,
    #             export_path=f"model_registry/{export_name}.keras"
    #         )
    #         m.plot_history(model_history)
    #         m.measure_metrics()
    #         m.predict_img("datasets/dataset_vers2/organic/organic_1.jpg")
    #         m.random_preds(3, 6)
    #         m.random_preds(8, 4)

    # ================================================================ SETTING UP BAR GRAPHS ================================================================

    # # set up the metrics for the model
    # all_metrics = {}
    # model_classes = [MobileNetV3Model, NASNetMobileModel] # change this as needed
    # image_sizes = [100, 125, 150, 175, 200, 225, 250, 275, 300]
    # for model_class in model_classes:
    #     model_metrics = {}
    #     for image_size in image_sizes:
    #         img_size = (image_size, image_size)
    #         export_name = f"{model_class.__name__}_imgsize_{img_size}"
    #         # logger = Logger(export_name)
    #         dl = DatasetManager(DATASET_DIR, img_size)
    #         m = model_class(dl)
    #         dl.load_data()
    #         dl.split_data(60, 20, 20)
    #         m.load_model_from_file(f"model_registry/{export_name}.keras")

    #         metrics_by_category = {}
    #         for cat in dl.get_categories():
    #             correct, wrong = m.metrics_for_category(cat)
    #             total = correct + wrong
    #             metrics_by_category[cat] = {
    #                 "correct": correct,
    #                 "wrong": wrong,
    #                 "accuracy": correct / total * 100,
    #             }
    #         model_metrics[img_size] = metrics_by_category
    #     all_metrics[model_class.__name__] = model_metrics

    # # create the bar chart: categories on x-axis, grouped bars for each image size
    # logger = Logger("bar_graphs")
    # for model_class in model_classes:
    #     model_metrics = all_metrics[model_class.__name__]
    #     categories = list(next(iter(model_metrics.values())).keys())
    #     img_sizes = list(model_metrics.keys())
    #     n_categories = len(categories)
    #     n_sizes = len(img_sizes)

    #     x = np.arange(n_categories)
    #     width = 0.8 / n_sizes

    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     for i, img_size in enumerate(img_sizes):
    #         accuracies = [model_metrics[img_size][cat]["accuracy"] for cat in categories]
    #         offset = (i - n_sizes / 2 + 0.5) * width
    #         ax.bar(x + offset, accuracies, width, label=f"{img_size[0]}×{img_size[1]}")

    #     ax.set_xlabel("Category")
    #     ax.set_ylabel("Accuracy (%)")
    #     ax.set_title(f"Accuracy by Category and Image Size (for {model_class.__name__})")
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(categories, rotation=45, ha="right")
    #     ax.legend(title="Image size")
    #     ax.set_ylim(0, 105)
    #     plt.tight_layout()
    #     logger.save_figure(f"{model_class.__name__}_bar_graph.png")





    # TEST 2: testing which minimum dimension to use for filtering out dataset images before the model ingests them
    # ================================================================ TRAINING THE MODELS WITH MINIMUM DIMENSIONS ================================================================
    # DATASET_DIR = "datasets/dataset_vers2/"
    # model_classes = [MobileNetV3Model]
    # image_sizes = [224]
    # min_dims = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]
    
    # for model_class in model_classes:
    #     for image_dim in image_sizes: # decided on 224x224 for the image size (it works the best)
    #         img_size = (image_dim, image_dim)
    #         for min_dim in min_dims:
    #             export_name = f"{model_class.__name__}_imgsize_{img_size}_skipdim_({min_dim})"
    #             logger = Logger(export_name)
    #             dl = DatasetManager(DATASET_DIR, img_size, logger=logger)
    #             m = model_class(dl, logger=logger)
    #             dl.load_data(min_dim=min_dim)
    #             dl.split_data(60, 20, 20)
    #             model_history = m.train_model(
    #                 epochs=25,
    #                 batch_size=32,
    #                 export_path=f"model_registry/{export_name}.keras"
    #             )
    #             m.plot_history(model_history)
    #             m.measure_metrics()
    #             m.predict_img("datasets/dataset_vers2/organic/organic_1.jpg")
    #             m.random_preds(3, 6)
    #             m.random_preds(8, 4)

    # ================================================================ LOAD MODELS AND CREATE BAR GRAPHS ================================================================
    # Load pre-trained models (no training), collect metrics, create bar charts
    # DATASET_DIR = "datasets/dataset_vers2/"
    # model_classes = [MobileNetV3Model]
    # image_sizes = [224]
    # min_dims = [10*i for i in range(1, 21)]

    # all_metrics = {}  # (model_name, img_size) -> {min_dim: {cat: {correct, wrong, accuracy}}}
    # all_pct_removed = {}  # (model_name, img_size) -> {min_dim: {cat: pct_removed}}
    # for model_class in model_classes:
    #     for image_dim in image_sizes:
    #         img_size = (image_dim, image_dim)
    #         all_metrics[(model_class.__name__, img_size)] = {}
    #         all_pct_removed[(model_class.__name__, img_size)] = {}
    #         for min_dim in min_dims:
    #             export_name = f"{model_class.__name__}_imgsize_{img_size}_skipdim_({min_dim})"
    #             dl = DatasetManager(DATASET_DIR, img_size)
    #             m = model_class(dl)
    #             dl.load_data(min_dim=min_dim)
    #             dl.split_data(60, 20, 20)
    #             m.load_model_from_file(f"model_registry/{export_name}.keras")

    #             metrics_by_category = {}
    #             for cat in dl.get_categories():
    #                 correct, wrong = m.metrics_for_category(cat)
    #                 total = correct + wrong
    #                 metrics_by_category[cat] = {
    #                     "correct": correct,
    #                     "wrong": wrong,
    #                     "accuracy": (correct / total * 100) if total > 0 else 0.0,
    #                 }
    #             all_metrics[(model_class.__name__, img_size)][min_dim] = metrics_by_category

    #             # track % images removed per category (for skip-dimension graphs)
    #             pct_removed_by_category = {}
    #             for cat in dl.get_categories():
    #                 loaded = len(dl.data_x[cat])
    #                 skipped = dl.skipped_by_category.get(cat, 0)
    #                 total = loaded + skipped
    #                 pct_removed_by_category[cat] = (skipped / total * 100) if total > 0 else 0.0
    #             all_pct_removed[(model_class.__name__, img_size)][min_dim] = pct_removed_by_category

    # # ================================================================ BAR CHARTS: ACCURACY BY CATEGORY AND MIN DIM ================================================================
    # # one graph per category; x-axis = min_dim, bars = accuracy for that category
    # bar_logger = Logger("bar_graphs_min_dim")
    # for (model_name, img_size), min_dim_metrics in all_metrics.items():
    #     categories = list(next(iter(min_dim_metrics.values())).keys())
    #     min_dim_values = sorted(min_dim_metrics.keys())

    #     for cat in categories:
    #         accuracies = [min_dim_metrics[min_dim][cat]["accuracy"] for min_dim in min_dim_values]
    #         x = np.arange(len(min_dim_values))
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.bar(x, accuracies, color="steelblue", alpha=0.8)
    #         ax.set_xlabel("Min dimension (in pixels)")
    #         ax.set_ylabel("Accuracy (%)")
    #         ax.set_title(f"Accuracy by Min Dimension — \"{cat}\" Category ({model_name}, {img_size[0]}×{img_size[1]})")
    #         ax.set_xticks(x)
    #         ax.set_xticklabels([f"{d}" for d in min_dim_values], rotation=45, ha="right")
    #         y_min, y_max = min(accuracies) - 10, max(accuracies) + 10
    #         ax.set_ylim(max(0, y_min), min(100, y_max))
    #         plt.tight_layout()
    #         bar_logger.save_figure(f"{model_name}_imgsize_{img_size}_accuracy_{cat}.png")

    # # ================================================================ BAR CHARTS: % IMAGES REMOVED PER CATEGORY ================================================================
    # # one graph per category; x-axis = min_dim, bars = % removed for that category
    # for (model_name, img_size), min_dim_pct in all_pct_removed.items():
    #     categories = list(next(iter(min_dim_pct.values())).keys())
    #     min_dim_values = sorted(min_dim_pct.keys())

    #     for cat in categories:
    #         pct_removed = [min_dim_pct[min_dim][cat] for min_dim in min_dim_values]
    #         x = np.arange(len(min_dim_values))
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         for i, (min_dim, pct) in enumerate(zip(min_dim_values, pct_removed)):
    #             ax.bar(x[i], pct, width=0.8, color="steelblue", alpha=0.8, label=f"min={min_dim}")
    #         ax.set_xlabel("Min dimension (in pixels)")
    #         ax.set_ylabel("% Images Removed")
    #         ax.set_title(f"% Images Removed by Min Dimension — \"{cat}\" Category ({model_name}, {img_size[0]}×{img_size[1]})")
    #         ax.set_xticks(x)
    #         ax.set_xticklabels([f"{d}" for d in min_dim_values], rotation=45, ha="right")
    #         ax.set_ylim(0, 100)
    #         plt.tight_layout()
    #         bar_logger.save_figure(f"{model_name}_imgsize_{img_size}_pct_removed_{cat}.png")





    # TEST 3: testing the threshold of accuracies between the top two predicted categories, for when to reject
    # ================================================================ TRAINING THE MODELS ================================================================
    # DATASET_DIR = "datasets/dataset_vers2/"
    # num_iters = 10
    # model_classes = [MobileNetV3Model]
    # image_sizes = [224]
    
    # for model_class in model_classes:
    #     for image_dim in image_sizes: # decided on 224x224 for the image size (it works the best)
    #         img_size = (image_dim, image_dim)
    #         for iter in range(num_iters): # train this many models to test on
    #             export_name = f"{model_class.__name__}_imgsize_{img_size}_iter{iter}"
    #             logger = Logger(export_name)
    #             dl = DatasetManager(DATASET_DIR, img_size, logger=logger)
    #             m = model_class(dl, logger=logger)
    #             dl.load_data()
    #             dl.split_data(60, 20, 20)
    #             model_history = m.train_model(
    #                 epochs=25,
    #                 batch_size=32,
    #                 export_path=f"model_registry/{export_name}.keras"
    #             )
    #             m.plot_history(model_history)
    #             m.measure_metrics()
    #             m.random_preds(3, 6)
    #             m.random_preds(8, 4)

    # ================================================================ OBSERVING THE TOP TWO PREDICTIONS ================================================================
    # add top two predictions into arrays (correct and wrong)
    # DATASET_DIR = "datasets/dataset_vers2/"
    # num_iters = 10

    # model_classes = [MobileNetV3Model]
    # image_sizes = [224]
    # for model_class in model_classes:
    #     for image_dim in image_sizes:
    #         img_size = (image_dim, image_dim)
    #         export_name = f"{model_class.__name__}_imgsize_{img_size}"
    #         correct_predictions = []
    #         wrong_predictions = []
    #         for iter in range(num_iters): # train this many models to test on
    #             print("Getting predictions for iter", iter)
    #             dl = DatasetManager(DATASET_DIR, img_size)
    #             m = model_class(dl)
    #             dl.load_data()
    #             dl.split_data(60, 20, 20)
    #             m.load_model_from_file(f"model_registry/{export_name}_iter{iter}.keras")

    #             predictions, pred_labels = m.predict_img_batch(dl.test_data_x)
    #             y_true = np.argmax(dl.test_data_y, axis=1)
    #             true_labels = [dl.get_category_name_from_index(i) for i in y_true]
    #             test_paths = dl.test_data_paths if hasattr(dl, 'test_data_paths') else [None] * len(true_labels)

    #             for i in range(len(true_labels)):
    #                 top2_idx = np.argsort(predictions[i])[-2:][::-1]
    #                 top2 = [(dl.get_category_name_from_index(j), float(predictions[i][j])) for j in top2_idx]
    #                 entry = {
    #                     "path": test_paths[i] if i < len(test_paths) else None,
    #                     "true_label": true_labels[i],
    #                     "pred_label": pred_labels[i],
    #                     "top2": top2,
    #                 }
    #                 if true_labels[i] == pred_labels[i]:
    #                     correct_predictions.append(entry)
    #                 else:
    #                     wrong_predictions.append(entry)

    #         # then, compute diffs and create graphs for each category
    #         print("Computing diffs and creating graphs")
    #         bar_logger = Logger(f"top_two_preds_{export_name}")
            
    #         correct_diffs = {cat: [] for cat in dl.get_categories()}
    #         wrong_diffs = {cat: [] for cat in dl.get_categories()}
    #         for pred_arr, pred_diffs in zip(
    #             [correct_predictions, wrong_predictions],
    #             [correct_diffs, wrong_diffs]
    #         ):
    #             for pred in pred_arr:
    #                 diff = abs(pred['top2'][0][1] - pred['top2'][1][1])
    #                 pred_diffs[pred['true_label']].append(diff)

    #         for cat in dl.get_categories():
    #             print("Creating graph for", cat)
    #             fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #             for ax, diffs, title in zip(
    #                 axes,
    #                 [correct_diffs[cat], wrong_diffs[cat]],
    #                 ["Correct", "Wrong"],
    #             ):
    #                 if len(diffs) > 0:
    #                     ax.hist(diffs, bins=20, color="steelblue", alpha=0.8, edgecolor="white")
    #                 ax.set_xlabel("Top2 diff (confidence gap)")
    #                 ax.set_ylabel("Count")
    #                 ax.set_title(f"{cat} — {title} (n={len(diffs)})")
    #             fig.suptitle(f"Distribution of Top2 Diff — {cat} (Sum of {num_iters} models)")
    #             plt.tight_layout()
    #             bar_logger.save_figure(f"top2_diff_dist_{cat}.png")

    #         # uncertainty cutoff analysis: for each cutoff, print %correct / %wrong / %uncertain per category
    #         all_predictions = correct_predictions + wrong_predictions
    #         cutoffs = [round(n * 0.05, 2) for n in range(21)]  # 0, 0.05, 0.1, ..., 1.0
    #         bar_logger.log_message(f"\n=== Uncertainty cutoff analysis ({num_iters} models) ===")
    #         for cat in dl.get_categories():
    #             cat_preds = [p for p in all_predictions if p["true_label"] == cat]
    #             if len(cat_preds) == 0:
    #                 continue
    #             pct_correct = []
    #             pct_wrong = []
    #             pct_uncertain = []
    #             for cutoff in cutoffs:
    #                 correct_confident = sum(1 for p in cat_preds if p["true_label"] == p["pred_label"] and abs(p["top2"][0][1] - p["top2"][1][1]) >= cutoff)
    #                 wrong_confident = sum(1 for p in cat_preds if p["true_label"] != p["pred_label"] and abs(p["top2"][0][1] - p["top2"][1][1]) >= cutoff)
    #                 uncertain = sum(1 for p in cat_preds if abs(p["top2"][0][1] - p["top2"][1][1]) < cutoff)
    #                 total = len(cat_preds)
    #                 pct_correct.append(correct_confident / total * 100)
    #                 pct_wrong.append(wrong_confident / total * 100)
    #                 pct_uncertain.append(uncertain / total * 100)
    #             bar_logger.log_message(f"\n--- {cat} (n={len(cat_preds)}) ---")
    #             bar_logger.log_message("Cutoff | % Correct | % Wrong | % Uncertain")
    #             for i, c in enumerate(cutoffs):
    #                 bar_logger.log_message(f"{c:5.2f} | {pct_correct[i]:8.1f} | {pct_wrong[i]:7.1f} | {pct_uncertain[i]:11.1f}")
    #             x = np.arange(len(cutoffs))
    #             width = 0.25
    #             fig, ax = plt.subplots(figsize=(14, 6))
    #             ax.bar(x - width, pct_correct, width, label="% Correct", color="green", alpha=0.8)
    #             ax.bar(x, pct_wrong, width, label="% Wrong", color="red", alpha=0.8)
    #             ax.bar(x + width, pct_uncertain, width, label="% Uncertain", color="gray", alpha=0.8)
    #             ax.set_xlabel("Uncertainty cutoff (top2 diff <= ...)")
    #             ax.set_ylabel("Percentage (%)")
    #             ax.set_title(f"# Predictions Correct / Wrong / Uncertain by cutoff - {cat} (Sum of {num_iters} models)")
    #             ax.set_xticks(x)
    #             ax.set_xticklabels([str(c) for c in cutoffs])
    #             ax.legend()
    #             plt.tight_layout()
    #             bar_logger.save_figure(f"uncertainty_cutoff_{cat}.png")

    #         # overall (all categories combined)
    #         pct_correct = []
    #         pct_wrong = []
    #         pct_uncertain = []
    #         total = len(all_predictions)
    #         for cutoff in cutoffs:
    #             correct_confident = sum(1 for p in all_predictions if p["true_label"] == p["pred_label"] and abs(p["top2"][0][1] - p["top2"][1][1]) >= cutoff)
    #             wrong_confident = sum(1 for p in all_predictions if p["true_label"] != p["pred_label"] and abs(p["top2"][0][1] - p["top2"][1][1]) >= cutoff)
    #             uncertain = sum(1 for p in all_predictions if abs(p["top2"][0][1] - p["top2"][1][1]) < cutoff)
    #             pct_correct.append(correct_confident / total * 100)
    #             pct_wrong.append(wrong_confident / total * 100)
    #             pct_uncertain.append(uncertain / total * 100)
    #         bar_logger.log_message(f"\n--- Overall (n={total}) ---")
    #         bar_logger.log_message("Cutoff | % Correct | % Wrong | % Uncertain")
    #         for i, c in enumerate(cutoffs):
    #             bar_logger.log_message(f"{c:5.2f} | {pct_correct[i]:8.1f} | {pct_wrong[i]:7.1f} | {pct_uncertain[i]:11.1f}")
    #         x = np.arange(len(cutoffs))
    #         width = 0.25
    #         fig, ax = plt.subplots(figsize=(14, 6))
    #         ax.bar(x - width, pct_correct, width, label="% Correct", color="green", alpha=0.8)
    #         ax.bar(x, pct_wrong, width, label="% Wrong", color="red", alpha=0.8)
    #         ax.bar(x + width, pct_uncertain, width, label="% Uncertain", color="gray", alpha=0.8)
    #         ax.set_xlabel("Uncertainty cutoff (top2 diff <= ...)")
    #         ax.set_ylabel("Percentage (%)")
    #         ax.set_title(f"Overall Predictions Correct / Wrong / Uncertain by cutoff (Sum of {num_iters} models)")
    #         ax.set_xticks(x)
    #         ax.set_xticklabels([str(c) for c in cutoffs])
    #         ax.legend()
    #         plt.tight_layout()
    #         bar_logger.save_figure("uncertainty_cutoff_overall.png")
            