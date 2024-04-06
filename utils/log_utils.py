import os

inference_rules = ['selective', 'climbing', 'jumping', 'maxcoverage', 'darts', 'maxlikelihood', 'maxlikelihood_all']
threshold_keys = ['opt_theta_' + i for i in inference_rules]
hAUROC_keys = ['hAUROC_flat'] + ['hAUROC_' + i for i in inference_rules]
haurc_keys = [f'hAURC_{i}_01' for i in inference_rules] + \
             ['hAURC_' + i + '_hier' for i in inference_rules] +\
             ['IMP_hAURC_' + i + '_01' for i in inference_rules[1:]] +\
             ['IMP_hAURC_' + i + '_hier' for i in inference_rules[1:]] 
hAccuracy_keys = ['hAccuracy_flat'] + ['hAccuracy_' + i for i in inference_rules]
marginal_cov_keys = ['marginal_coverage_' + i for i in inference_rules]
hsc_keys = threshold_keys + hAUROC_keys + haurc_keys + hAccuracy_keys + marginal_cov_keys 

headers = ['Architecture', 'Accuracy', 'Accuracy_TS', 'AUROC', 'AUROC_TS', 'ECE_15', 'ECE_15_TS', 'MCE_15', 'MCE_15_TS',
            'Gamma Correlation', 'Gamma Correlation_TS', 'Parameters #', 'AURC', 'AURC_TS',
            'Confidence Mean', 'Confidence Mean_TS', 'Confidence Median', 'Confidence Median_TS', 'Confidence GINI',
            'Confidence GINI_TS', 'Confidence Variance', 'Confidence Variance_TS', 'A-ECE', 'A-ECE_TS', 'A-MCE',
            'A-MCE_TS', 'Parameters # Trainable', 'ECE_10', 'ECE_10_TS', 'MCE_10', 'MCE_10_TS', 'ECE_30',
            'ECE_30_TS', 'MCE_30', 'MCE_30_TS', 'ECE_50', 'ECE_50_TS', 'MCE_50', 'MCE_50_TS', 'input size',
            'Temperature', 'Coverage_for_Accuracy_95', 'Coverage_for_Accuracy_95_nonstrict',
            'Coverage_for_Accuracy_95_TS', 'Coverage_for_Accuracy_95_nonstrict_TS', 'Coverage_for_Accuracy_99',
            'Coverage_for_Accuracy_99_nonstrict', 'Coverage_for_Accuracy_99_TS',
            'Coverage_for_Accuracy_99_nonstrict_TS', 'Coverage_for_Accuracy_98', 'Coverage_for_Accuracy_98_nonstrict',
            'Coverage_for_Accuracy_98_TS', 'Coverage_for_Accuracy_98_nonstrict_TS', 'Coverage_for_Accuracy_97',
            'Coverage_for_Accuracy_97_nonstrict', 'Coverage_for_Accuracy_97_TS',
            'Coverage_for_Accuracy_97_nonstrict_TS', 'Coverage_for_Accuracy_96', 'Coverage_for_Accuracy_96_nonstrict',
            'Coverage_for_Accuracy_96_TS', 'Coverage_for_Accuracy_96_nonstrict_TS', 'EAURC', 'EAURC_TS', ] + \
            ['hCal_leaves', 'hCal_leaves_TS', 'alpha', 'alpha_TS'] + \
            hsc_keys + [k+'_TS' for k in hsc_keys]

class Logger:

    def __init__(self, file_name, headers, overwrite):
        self.log_file_name = file_name
        self.headers = headers
        if overwrite and os.path.isfile(file_name):
            os.remove(file_name)
        if (os.path.isfile(file_name) and os.path.getsize(file_name) > 0) or headers is None:
            return
        msg = ''
        for header in self.headers:
            msg += f'{header},'

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(self.log_file_name, 'a') as f:
            f.write(f'{msg[:-1]}\n')

    def log(self, msg_dict):
        with open(self.log_file_name, 'a') as f:
            msg = ''
            for header in self.headers:
                if header not in msg_dict:
                    if 'TS' not in header:
                        print(f'WARNING {header} was not logged')
                    cell_content = None
                else:
                    cell_content = msg_dict[header]
                if type(cell_content) is float:
                    msg += f'{cell_content: .5f},'
                else:
                    msg += f'{cell_content},'

            print(msg)
            f.write(f'{msg[:-1]}\n')

    def log_msg(self, msg):
        with open(self.log_file_name, 'a') as f:
            f.write(f'{msg}\n')

    def get_df(self):
        import pandas as pd
        return pd.read_csv(self.log_file_name)


# utility for timing execution
class Timer:
    import time
    timer = time.time

    def __init__(self, msg='time elapsed', print_human_readable=False, file_name=None):
        self._timer = self.timer
        self._start = 0
        self._end = 0
        self.msg = msg
        self.print_human_readable = print_human_readable
        self.file_name = file_name
        if file_name is not None:
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

    def __enter__(self):
        self._start = self._timer()
        return self.start()

    def __exit__(self, a, b, c):
        self.end()

    def start(self):
        self._start = self._timer()
        return self

    def end(self):
        self._end = self._timer()
        period = self._end - self._start
        if period > 1:
            _time_str = get_human_readable_time(period)
        else:
            _time_str = str(period) + ' secs'
        if self.file_name is not None:
            with open(self.file_name, mode='a') as f:
                print(f'{self.msg} {_time_str}', file=f)
        else:
            print(f'{self.msg} {_time_str}')

    def get_time(self):
        return self._timer() - self._start


def get_human_readable_time(secs):
    time_str = ''
    for unit, sec_in_unit in zip(['day(s)', 'hr(s)', 'mints', 'secs'], [3600 * 24, 3600, 60, 1]):

        time_in_unit = secs // sec_in_unit
        secs %= sec_in_unit

        if time_in_unit >= 1 or unit == 'secs':
            time_str += f'{time_in_unit} {unit} '

    return time_str[:-1]
