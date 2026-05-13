"""
Play Hardball: Hard-to-Easy Curriculum Learning for Mitigating Spurious Correlations.
Written by Aiyang Han, PARNEC
https://github.com/hannaiiyanggit/HECL
"""

import numpy as np

class local_log():
    def __init__(self, args):
        self.dic = {}
        self.args = args
        self.keys = []

    def add_metric(self, name):
        if name not in self.keys:
            self.dic[name] = [[]]
            self.keys.append(name)

    def refresh(self):
        for key in list(self.dic.keys()):
            self.dic[key].append([])

    def update(self, log_dict):
        for key in list(log_dict.keys()):
            self.add_metric(key)
            self.dic[key][-1].append(log_dict[key])

    def single_summary(self):
        for key in list(self.dic.keys()):
            print(f"{key}: {self.dic[key][-1][-1]*100:.2f}")

    def summary(self):
        results_str = []
        summary_data = {}

        header = f"{'--' * 5} {self.args.method} | {self.args.dataset} {'--' * 5}"
        results_str.append(header)
        results_str.append(str(self.args))

        print(f"\nRun Summary:\n{header}")

        for key, values in self.dic.items():
            last_results = [v[-1] for v in values]
            summary_data[key] = last_results

            mu, std = np.mean(last_results) * 100, np.std(last_results) * 100

            stat_line = f"{key}:  mean: {mu:.2f}  std: {std:.2f}"
            raw_data_line = f"Raw: {last_results}"

            print(stat_line)
            results_str.extend([stat_line, raw_data_line])

        with open("output.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(results_str) + "\n\n")