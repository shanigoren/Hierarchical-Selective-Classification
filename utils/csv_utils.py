

import csv

def save_results_csv(results, model_name, exp_name, metrics, thetas, inf_rules_names, alpha=None, n=None):
        if exp_name == 'inference_rules':
            with open(f'results/{exp_name}/{model_name}.csv', 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['inference_rule', 'theta']+metrics)
                writer.writeheader()
                for theta in thetas:
                    for inf_rule in inf_rules_names:
                        writer.writerow({'inference_rule': inf_rule, 
                                        'theta': theta, 
                                        **{metric: results[theta][inf_rule][metric] for metric in metrics}})

        if exp_name == 'thresholds':
            with open(f'results/{exp_name}/{model_name}_alpha_{alpha}_n_{n}.csv', 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['rep', 'inference_rule', 'theta']+metrics)
                writer.writeheader()
                for rep in results:
                    for inf_rule in thetas:
                        theta = thetas[inf_rule].item()
                        writer.writerow({'rep': rep,
                                        'inference_rule': inf_rule, 
                                        'theta': results[rep][inf_rule]['theta'], 
                                        **{metric: results[rep][inf_rule][metric] for metric in metrics}})