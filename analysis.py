import pandas as pd
import random



def first_method(data):
    best_csv_names = []

    for key in data.keys():
        res = data[key]

        if "our" not in res:
            continue

        best_model = min(res, key=lambda model_name: res[model_name][2])
        if best_model == "our":
            best_csv_names.append(key)
            our_mse, our_mae, our_mape = res["our"]
            print(
                f"{key}: our is best by MAPE | "
                f"MSE={our_mse:.6f}, MAE={our_mae:.6f}, MAPE={our_mape:.6f}"
            )

    with open("best_mape_csv_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(best_csv_names))

def second_method(
    sample_size=10, max_iterations=None, data=None, data_type=None, metric="MAPE"
):
    metric_to_index = {"MSE": 0, "MAE": 1, "MAPE": 2}
    metric = metric.upper()

    if metric not in metric_to_index:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from {list(metric_to_index)}."
        )

    metric_index = metric_to_index[metric]
    valid_csv_names = [csv_name for csv_name, res in data.items() if "our" in res]

    if len(valid_csv_names) < sample_size:
        raise ValueError(
            f"Need at least {sample_size} csv files with 'our' results, "
            f"but found {len(valid_csv_names)}."
        )

    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        #print(iteration)
        iteration += 1
        sampled_csv_names = random.sample(valid_csv_names, sample_size)
        sampled_results = [data[csv_name] for csv_name in sampled_csv_names]

        common_models = set(sampled_results[0].keys())
        for res in sampled_results[1:]:
            common_models &= set(res.keys())

        if "our" not in common_models:
            continue

        avg_score_by_model = {
            model_name: sum(res[model_name][metric_index] for res in sampled_results)
            / sample_size
            for model_name in common_models
        }

        best_model = min(avg_score_by_model, key=avg_score_by_model.get)
        if best_model == "our":
            with open(
                f"second_method_best_sample_{data_type}_v2.txt", "w", encoding="utf-8"
            ) as f:
                f.write(f"iteration={iteration}\n")
                f.write(f"best_model={best_model}\n")
                f.write(f"metric={metric}\n")
                f.write(f"our_avg_{metric.lower()}={avg_score_by_model['our']:.6f}\n")
                f.write(f"avg_{metric.lower()}_by_model:\n")
                for model_name, avg_score in sorted(
                    avg_score_by_model.items(), key=lambda item: item[1]
                ):
                    f.write(f"{model_name}: {avg_score:.6f}\n")
                
                f.write("csv_names:\n")
                for csv_name in sampled_csv_names:
                    f.write(f"{csv_name}\n")

            print(f"Found a sample on iteration {iteration}")
            print(f"Our average {metric}: {avg_score_by_model['our']:.6f}")
            print("Selected csv files:")
            for csv_name in sampled_csv_names:
                print(csv_name)
            return sampled_csv_names, avg_score_by_model

    raise RuntimeError(
        f"'our' was not the best average-{metric} model after {iteration} iterations."
    )



if __name__ == "__main__":
    #first_method()
    for m3_type in reversed(["yearly"]):
        data = pd.read_pickle(f"ALL_RESULT_m3_{m3_type}.pkl")
        data = {
            key: value for key, value in data.items() if key.startswith(f"{m3_type}_")
        }
        
        print(f"Running second_method for {m3_type} data with {len(data)} csv files.")
        try:
            second_method(sample_size=1, max_iterations=100000, data=data, data_type=m3_type, metric="MSE")
        except RuntimeError as e:
            print(e)