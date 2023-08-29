import ipdb
import pandas as pd
import numpy as np
import tqdm
import collections
import sigfig


def main_tables(mean, spread, aggregate_mean, aggregate_spread):
    main_autoencoders = {k: k for k in [
        'AE',
        'β-VAE',
        'β-TCVAE',
        'BioAE',
        'QLAE (ours)',
    ]}
    main_autoencoders['β-VAE'] = r'$\beta$-VAE'
    main_autoencoders['β-TCVAE'] = r'$\beta$-TCVAE'

    main_gans = {k: k for k in [
        'InfoWGAN-GP',
        'QLInfoWGAN-GP (ours)',
    ]}

    subtables = []
    for model_class in [main_autoencoders, main_gans]:
        subtable = pd.DataFrame()
        for model in model_class.keys():
            row = collections.OrderedDict()
            for dataset in ['aggregate', 'shapes3d', 'mpi3d', 'falcor3d', 'isaac3d']:
                for metric in ['infom', 'infoe', 'infoc']:
                    if dataset == 'aggregate':
                        mean_value = aggregate_mean.loc[model, metric]
                        spread_value = aggregate_spread.loc[model, metric]
                    else:
                        mean_value = mean.loc[model, dataset, metric]
                        spread_value = spread.loc[model, dataset, metric]
                    row[f'{dataset}_{metric}'] = mean_value, spread_value
            subtable = subtable.append(row, ignore_index=True)
        subtables.append(process_main_table(subtable, model_class.values()))

    table = subtables[0][subtables[0].find('AE'):subtables[0].find(r'\bottomrule')]
    table += ' \\midrule\n'
    table += subtables[1][subtables[1].find('InfoWGAN-GP'):subtables[1].find(r'\bottomrule') + len(r'\bottomrule')]

    with open(f'results/table_main.tex', 'w') as f:
        f.write(table)

    ablation_autoencoders = {k: k for k in [
        'QLAE (ours)',
        'QLAE w/ global codebook',
        'QLAE w/o weight decay',
        'VQ-VAE w/ weight decay',
        'VQ-VAE w/o weight decay',
    ]}

    ablation_gans = {k: k for k in [
        'QLInfoWGAN-GP (ours)',
        'QLInfoWGAN-GP w/o weight decay'
    ]}

    subtables = []
    for model_class in [ablation_autoencoders, ablation_gans]:
        subtable = pd.DataFrame()
        for model in model_class.keys():
            row = collections.OrderedDict()
            for dataset in ['aggregate', 'shapes3d', 'mpi3d', 'falcor3d', 'isaac3d']:
                for metric in ['infom', 'infoe', 'infoc']:
                    if dataset == 'aggregate':
                        mean_value = aggregate_mean.loc[model, metric]
                        spread_value = aggregate_spread.loc[model, metric]
                    else:
                        mean_value = mean.loc[model, dataset, metric]
                        spread_value = spread.loc[model, dataset, metric]
                    row[f'{dataset}_{metric}'] = mean_value, spread_value
            subtable = subtable.append(row, ignore_index=True)
        subtables.append(process_main_table(subtable, model_class.values()))

    table = subtables[0][subtables[0].find('QLAE (ours)'):subtables[0].find(r'\bottomrule')]
    table += ' \\midrule\n'
    table += subtables[1][subtables[1].find('QLInfoWGAN-GP (ours)'):subtables[1].find(r'\bottomrule') + len(r'\bottomrule')]

    with open(f'results/table_ablations.tex', 'w') as f:
        f.write(table)


def process_main_table(table, model_column):
    upper = table.applymap(lambda x: x[0] + x[1])
    lower = table.applymap(lambda x: x[0] - x[1])
    i_highest = upper.idxmax(axis=0)
    lower_of_highest = lower.lookup(i_highest, upper.columns)
    bold = upper >= lower_of_highest

    entries = table.applymap(lambda x: rf'{x[0]:.2f}')

    bolded = entries.copy()
    bolded[bold] = bolded[bold].applymap(lambda x: rf'\mathbf{{{x}}}')

    gray_columns = bolded.columns[bolded.columns.str.contains('infoc')]
    grayed = bolded.copy()
    grayed[gray_columns] = grayed[gray_columns].applymap(lambda x: rf'{{\color{{gray}} {x}}}')

    left_columns = grayed.columns[grayed.columns.str.contains('infom')]
    lefted = grayed.copy()
    lefted[left_columns] = lefted[left_columns].applymap(lambda x: rf'({x}')

    right_columns = grayed.columns[grayed.columns.str.contains('infoc')]
    righted = lefted.copy()
    righted[right_columns] = righted[right_columns].applymap(lambda x: rf'{x})')

    dollared = righted.applymap(lambda x: rf'${x}$')
    dollared.insert(0, 'model', model_column)
    return dollared.to_latex(escape=False, index=False, column_format='lcccccccccccccccc')


def full_tables(mean, spread, aggregate_mean, aggregate_spread):
    autoencoders = {k: k for k in [
        'AE',
        'β-VAE',
        'β-TCVAE',
        'BioAE',
        'QLAE (ours)',
        'QLAE w/ global codebook',
        'QLAE w/o weight decay',
        'VQ-VAE w/ weight decay',
        'VQ-VAE w/o weight decay',
    ]}
    autoencoders['β-VAE'] = r'$\beta$-VAE'
    autoencoders['β-TCVAE'] = r'$\beta$-TCVAE'

    gans = {k: k for k in [
        'InfoWGAN-GP',
        'QLInfoWGAN-GP (ours)',
        'QLInfoWGAN-GP w/o weight decay'
    ]}

    for dataset in ['shapes3d', 'mpi3d', 'falcor3d', 'isaac3d', 'aggregate']:
        subtables = []
        for model_class in [autoencoders, gans]:
            subtable = pd.DataFrame()
            for model in model_class.keys():
                row = collections.OrderedDict()
                for metric in ['infom', 'infoe', 'infoc', 'psnr']:
                    if dataset == 'aggregate':
                        mean_value = aggregate_mean.loc[model, metric]
                        spread_value = aggregate_spread.loc[model, metric]
                    else:
                        mean_value = mean.loc[model, dataset, metric]
                        spread_value = spread.loc[model, dataset, metric]
                    row[f'{dataset}_{metric}'] = mean_value, spread_value
                subtable = subtable.append(row, ignore_index=True)

            subtables.append(process_full_table(subtable, model_class.values()))

        table = subtables[0][:subtables[0].find(r'\bottomrule')]
        table += ' \\midrule\n'
        table += subtables[1][subtables[1].find('InfoWGAN-GP'):]

        with open(f'results/table_{dataset}.tex', 'w') as f:
            f.write(table)


def process_full_table(table, model_column):
    upper = table.applymap(lambda x: x[0] + x[1])
    lower = table.applymap(lambda x: x[0] - x[1])
    i_highest = upper.idxmax(axis=0)
    lower_of_highest = lower.lookup(i_highest, upper.columns)
    bold = upper >= lower_of_highest

    sep = r' \pm '
    intervals = table.applymap(lambda x: rf'{sigfig.round(x[0], uncertainty=x[1], sep=sep)}')

    bolded = intervals.copy()
    bolded[bold] = bolded[bold].applymap(lambda x: rf'\mathbf{{{x}}}')

    dollared = bolded.applymap(lambda x: rf'${x}$')
    dollared.insert(0, 'model', model_column)
    return dollared.to_latex(escape=False, index=False, column_format='lcccc', header=[
        'model',
        r'$\text{InfoM} \uparrow$',
        r'$\text{InfoE} \uparrow$',
        r'$\text{InfoC} \uparrow$',
        r'$\text{PSNR (dB)} \uparrow$'
    ])


def main():
    t_values = {
        6: 2.447,
        27: 2.052
    }

    df = pd.read_csv('results/data.csv')

    mean = df.groupby(['model', 'dataset', 'metric']).mean().squeeze()
    standard_error = df.groupby(['model', 'dataset', 'metric']).apply(
        lambda x: np.std(x) / np.sqrt(len(x))).squeeze()
    dof = df.groupby(['model', 'dataset', 'metric']).apply(lambda x: len(x) - 1)
    t_values = dof.apply(lambda x: t_values[x])
    ci95 = t_values * standard_error

    aggregate_mean = df.groupby(['model', 'metric']).mean().squeeze()
    aggregate_standard_error = standard_error.groupby(['model', 'metric']).mean()
    aggregate_dof = df.groupby(['model', 'metric']).apply(lambda x: len(x) - 1)
    aggregate_t_values = aggregate_dof.apply(lambda x: t_values[x])
    aggregate_ci95 = aggregate_t_values * aggregate_standard_error

    full_tables(mean, ci95, aggregate_mean, aggregate_ci95)
    main_tables(mean, ci95, aggregate_mean, aggregate_ci95)


if __name__ == '__main__':
    main()
