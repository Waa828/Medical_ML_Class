import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def get_compound_data(df, compound_id):
    """
    Extract data for a specific compound and categorize by solvent system.
    """
    sub = df[df['COMPOUND_ID'] == compound_id].copy()

    # EA system: H + EA (no MeOH, no Et2O, no DCM)
    ea_data = sub[(sub['EA'] > 0) & (sub['MeOH'] == 0) & (sub['Et2O'] == 0) & (sub['DCM'] == 0)].copy()
    ea_data = ea_data[ea_data['H'] > 0]  # Exclude pure EA (H=0)
    if len(ea_data) > 0:
        ea_data['ratio'] = ea_data['EA'] / ea_data['H']
        ea_data['log_ratio'] = np.log10(ea_data['ratio'].replace(0, 1e-10))

    # MeOH system: DCM + MeOH (no EA, no Et2O)
    meoh_data = sub[(sub['MeOH'] > 0) & (sub['EA'] == 0) & (sub['Et2O'] == 0)].copy()
    if len(meoh_data) > 0:
        meoh_data['ratio'] = meoh_data['MeOH'] / meoh_data['DCM'].replace(0, 1e-10)
        meoh_data['log_ratio'] = np.log10(meoh_data['ratio'])

    # Et2O system: H + Et2O (no EA, no MeOH, no DCM)
    et2o_data = sub[(sub['Et2O'] > 0) & (sub['EA'] == 0) & (sub['MeOH'] == 0) & (sub['DCM'] == 0)].copy()
    et2o_data = et2o_data[et2o_data['H'] > 0]  # Exclude pure Et2O (H=0)
    if len(et2o_data) > 0:
        et2o_data['ratio'] = et2o_data['Et2O'] / et2o_data['H']
        et2o_data['log_ratio'] = np.log10(et2o_data['ratio'].replace(0, 1e-10))

    smiles = sub.iloc[0]['COMPOUND_SMILES'] if len(sub) > 0 else None
    name = sub.iloc[0]['COMPOUND_ENG_NAME'] if len(sub) > 0 else None

    return {
        'ea': ea_data,
        'meoh': meoh_data,
        'et2o': et2o_data,
        'smiles': smiles,
        'name': name
    }


def get_molecule_image(smiles, size=(300, 300)):
    """
    Generate molecule image from SMILES string using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=size, kekulize=True)
    return img


def add_molecule_to_ax(ax, smiles, position=(0.65, 0.15), size=0.3):
    """
    Add molecule structure image to an axes at specified position.
    position: (x, y) in axes coordinates
    size: size of the inset axes
    """
    img = get_molecule_image(smiles)
    # Create inset axes for molecule
    inset_ax = ax.inset_axes([position[0], position[1], size, size])
    inset_ax.imshow(img)
    inset_ax.axis('off')
    inset_ax.set_facecolor('none')


def plot_compound_row(ax_row, data_dict, compound_label, color_observed):
    """
    Plot a single row (one compound) with three solvent systems.
    ax_row: list of three axes [EA_ax, MeOH_ax, Et2O_ax]
    """
    solvent_systems = [
        ('ea', 'Log EA ratio', data_dict['ea']),
        ('meoh', 'Log MeOH ratio', data_dict['meoh']),
        ('et2o', 'Log Et₂O ratio', data_dict['et2o'])
    ]

    for idx, (system, xlabel, df_data) in enumerate(solvent_systems):
        ax = ax_row[idx]

        if len(df_data) == 0:
            ax.set_visible(False)
            continue

        x_data = df_data['log_ratio'].values
        y_data = df_data['Rf'].values

        # Sort by x for proper line plotting
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]

        # Plot line connecting observed points
        ax.plot(x_sorted, y_sorted, '-', color=color_observed, linewidth=1.5,
                label='Observed Rf curve')

        # Plot observed values as triangles
        ax.scatter(x_sorted, y_sorted, c=color_observed, marker='^', s=80,
                   edgecolors='darkred', linewidth=0.5, zorder=5, label='Observed Rf values')

        # Formatting
        ax.set_xlabel(xlabel, fontsize=10)
        if idx == 0:
            ax.set_ylabel('Rf', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(min(x_data) - 0.05, max(x_data) + 0.05)
        ax.grid(True, alpha=0.3)

        # Add legend to each subplot (positioned to avoid overlapping with molecule)
        ax.legend(loc='lower left', fontsize=7, framealpha=0.9)

        # Add molecule structure to ALL subplots
        if data_dict['smiles']:
            add_molecule_to_ax(ax, data_dict['smiles'], position=(0.68, 0.12), size=0.32)


def create_tlc_plots(df, compound_ids):
    """
    Parameters:
    -----------
    df : pandas.DataFrame
        The TLC dataset
    compound_ids : list
        List of compound IDs to plot (should be 3 compounds)
    """
    # labels for each compounds
    labels = ['A', 'B', 'C']

    # Colors for observed values
    colors = ['#DC143C', '#228B22', '#FF8C00']  # Crimson, Forest Green, Dark Orange

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.patch.set_facecolor('#E8E8E8')  # Light gray background like refer.jpg

    for row_idx, (compound_id, label, color) in enumerate(zip(compound_ids, labels, colors)):
        data_dict = get_compound_data(df, compound_id)
        plot_compound_row(axes[row_idx], data_dict, label, color)

        # Add compound label on the left side
        axes[row_idx, 0].annotate(f'{label}', xy=(-0.25, 0.95), xycoords='axes fraction',
                                  fontsize=14, fontweight='bold', va='center', ha='center',
                                  transform=axes[row_idx, 0].transAxes)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, wspace=0.3, hspace=0.3)
    return fig, axes


def main():
    # Load data
    df = pd.read_excel('TLC_dataset.xlsx')

    # Define compound IDs to plot (1, 3, 4)
    compound_ids = [1, 3, 4]

    # Create plots
    fig, axes = create_tlc_plots(df, compound_ids)

    # Save figure
    plt.savefig('tlc_plots.png', dpi=300, bbox_inches='tight', facecolor='#E8E8E8')
    plt.show()


if __name__ == "__main__":
    main()

